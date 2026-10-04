package zarr

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"slices"
	"sort"
	"strings"
)

// ErrNotFound is returned when a node does not exist.
var ErrNotFound = errors.New("zarr: node not found")

// ErrExists is returned when creating a node where one exists.
var ErrExists = errors.New("zarr: node exists")

// Node is an *Array or a *Group.
type Node interface {
	Path() string
	ZarrFormat() int
	Attributes() map[string]any
}

// CreateOptions tunes array and group creation.
type CreateOptions struct {
	Options
	// Overwrite replaces an existing node (deleting its contents);
	// otherwise creating over an existing node fails with ErrExists.
	Overwrite bool
}

// Group is a Zarr group.
type Group struct {
	store  Store
	path   string
	format int
	attrs  map[string]any
	raw    any        // ordered attributes
	doc    jsonObject // v3 zarr.json
	// consolidated maps member paths (relative to the group) to their
	// metadata, when the group has consolidated metadata.
	consolidated map[string]consolidatedEntry
	opts         *Options
}

type consolidatedEntry struct {
	format int
	array  bool
	doc    []byte // zarr.json or .zarray
	attrs  []byte // v2 .zattrs
}

func (g *Group) Path() string               { return g.path }
func (g *Group) ZarrFormat() int            { return g.format }
func (g *Group) Attributes() map[string]any { return g.attrs }

// Consolidated reports whether the group was opened with consolidated
// metadata, which then answers member lookups.
func (g *Group) Consolidated() bool { return g.consolidated != nil }

// ---------------------------------------------------------------------------
// Reading metadata

// nodeDocs reads the metadata of the node at p: v3 zarr.json wins over v2
// documents, as in zarr-python.
type nodeDocs struct {
	format   int
	nodeType string
	doc      []byte // zarr.json, .zarray or .zgroup
	attrs    []byte // v2 .zattrs
}

func readNode(ctx context.Context, store Store, p string) (*nodeDocs, error) {
	doc, ok, err := store.Get(ctx, joinKey(p, "zarr.json"), ByteRange{})
	if err != nil {
		return nil, err
	}
	if ok {
		var probe struct {
			ZarrFormat int    `json:"zarr_format"`
			NodeType   string `json:"node_type"`
		}
		if err := json.Unmarshal(pyCompat(doc), &probe); err != nil {
			return nil, fmt.Errorf("zarr: invalid %s: %w", joinKey(p, "zarr.json"), err)
		}
		return &nodeDocs{format: 3, nodeType: probe.NodeType, doc: doc}, nil
	}
	n := &nodeDocs{format: 2}
	attrs, ok, err := store.Get(ctx, joinKey(p, ".zattrs"), ByteRange{})
	if err != nil {
		return nil, err
	}
	if ok {
		n.attrs = attrs
	}
	if doc, ok, err = store.Get(ctx, joinKey(p, ".zarray"), ByteRange{}); err != nil {
		return nil, err
	} else if ok {
		n.nodeType, n.doc = "array", doc
		return n, nil
	}
	if doc, ok, err = store.Get(ctx, joinKey(p, ".zgroup"), ByteRange{}); err != nil {
		return nil, err
	} else if ok {
		n.nodeType, n.doc = "group", doc
		return n, nil
	}
	return nil, fmt.Errorf("%w: %q", ErrNotFound, p)
}

// Open opens the array or group at path.
func Open(ctx context.Context, store Store, p string, opts *Options) (Node, error) {
	p, err := normalizePath(p)
	if err != nil {
		return nil, err
	}
	n, err := readNode(ctx, store, p)
	if err != nil {
		return nil, err
	}
	return openNode(ctx, store, p, n, opts)
}

func openNode(ctx context.Context, store Store, p string, n *nodeDocs, opts *Options) (Node, error) {
	switch n.nodeType {
	case "array":
		return arrayFromDocs(store, p, n, opts)
	case "group":
		return groupFromDocs(ctx, store, p, n, opts)
	}
	return nil, fmt.Errorf("zarr: %q has node_type %q", p, n.nodeType)
}

// OpenArray opens the array at path in store.
func OpenArray(ctx context.Context, store Store, p string, opts *Options) (*Array, error) {
	node, err := Open(ctx, store, p, opts)
	if err != nil {
		return nil, err
	}
	a, ok := node.(*Array)
	if !ok {
		return nil, fmt.Errorf("zarr: %q is a group, not an array", p)
	}
	return a, nil
}

// OpenGroup opens the group at path in store, using its consolidated
// metadata if it has any.
func OpenGroup(ctx context.Context, store Store, p string, opts *Options) (*Group, error) {
	node, err := Open(ctx, store, p, opts)
	if err != nil {
		return nil, err
	}
	g, ok := node.(*Group)
	if !ok {
		return nil, fmt.Errorf("zarr: %q is an array, not a group", p)
	}
	return g, nil
}

func arrayFromDocs(store Store, p string, n *nodeDocs, opts *Options) (*Array, error) {
	var m *Metadata
	var err error
	if n.format == 2 {
		m, err = ParseMetadataV2(n.doc, n.attrs)
	} else {
		m, err = parseV3Metadata(n.doc)
	}
	if err != nil {
		return nil, fmt.Errorf("zarr: array %q: %w", p, err)
	}
	a, err := NewArrayFromMetadata(m, StoreChunks(store, p, m.ChunkKeyEncoding), opts)
	if err != nil {
		return nil, fmt.Errorf("zarr: array %q: %w", p, err)
	}
	a.store, a.path = store, p
	return a, nil
}

func groupFromDocs(ctx context.Context, store Store, p string, n *nodeDocs, opts *Options) (*Group, error) {
	g := &Group{store: store, path: p, format: n.format, opts: opts}
	var err error
	if n.format == 3 {
		o, err := parseOrdered(n.doc)
		if err != nil {
			return nil, fmt.Errorf("zarr: group %q: %w", p, err)
		}
		g.doc, _ = o.(jsonObject)
		var raw struct {
			Attributes   json.RawMessage `json:"attributes"`
			Consolidated *struct {
				Kind     string                     `json:"kind"`
				Metadata map[string]json.RawMessage `json:"metadata"`
			} `json:"consolidated_metadata"`
		}
		if err := json.Unmarshal(pyCompat(n.doc), &raw); err != nil {
			return nil, fmt.Errorf("zarr: group %q: %w", p, err)
		}
		if g.attrs, err = decodeAttributes(raw.Attributes); err != nil {
			return nil, err
		}
		g.raw, _ = g.doc.get("attributes")
		if c := raw.Consolidated; c != nil && c.Metadata != nil && len(c.Metadata) > 0 {
			g.consolidated = map[string]consolidatedEntry{}
			for k, v := range c.Metadata {
				var probe struct {
					NodeType string `json:"node_type"`
				}
				json.Unmarshal(v, &probe)
				g.consolidated[k] = consolidatedEntry{format: 3, array: probe.NodeType == "array", doc: v}
			}
		}
		return g, nil
	}
	if n.attrs != nil {
		if g.attrs, err = decodeAttributes(pyCompat(n.attrs)); err != nil {
			return nil, err
		}
		g.raw, _ = parseOrdered(n.attrs)
	} else {
		g.attrs = map[string]any{}
	}
	// Zarr v2 consolidated metadata lives in .zmetadata.
	zm, ok, err := store.Get(ctx, joinKey(p, ".zmetadata"), ByteRange{})
	if err != nil {
		return nil, err
	}
	if ok {
		var raw struct {
			Metadata map[string]json.RawMessage `json:"metadata"`
		}
		if err := json.Unmarshal(pyCompat(zm), &raw); err != nil {
			return nil, fmt.Errorf("zarr: invalid .zmetadata: %w", err)
		}
		g.consolidated = map[string]consolidatedEntry{}
		for k, v := range raw.Metadata {
			dir, file := path.Split(k)
			dir = strings.TrimSuffix(dir, "/")
			if dir == "" {
				continue // the group itself
			}
			e := g.consolidated[dir]
			e.format = 2
			switch file {
			case ".zarray":
				e.array, e.doc = true, v
			case ".zgroup":
				e.doc = v
			case ".zattrs":
				e.attrs = v
			}
			g.consolidated[dir] = e
		}
	}
	return g, nil
}

// ---------------------------------------------------------------------------
// Members

// Member opens a member (a child, or a descendant with a "/"-separated
// relative path). With consolidated metadata, only nodes it lists are
// found (as in zarr-python).
func (g *Group) Member(ctx context.Context, name string) (Node, error) {
	name, err := normalizePath(name)
	if err != nil || name == "" {
		return nil, fmt.Errorf("zarr: invalid member name %q", name)
	}
	full := joinKey(g.path, name)
	if g.consolidated != nil {
		e, ok := g.consolidated[name]
		if !ok || e.doc == nil {
			return nil, fmt.Errorf("%w: %q not in consolidated metadata", ErrNotFound, name)
		}
		n := &nodeDocs{format: e.format, nodeType: "group", doc: e.doc, attrs: e.attrs}
		if e.array {
			n.nodeType = "array"
		}
		node, err := openNode(ctx, g.store, full, n, g.opts)
		if sub, ok := node.(*Group); ok && err == nil {
			// Descendants come from the same consolidated metadata.
			sub.consolidated = map[string]consolidatedEntry{}
			for k, v := range g.consolidated {
				if rest, ok := strings.CutPrefix(k, name+"/"); ok {
					sub.consolidated[rest] = v
				}
			}
		}
		return node, err
	}
	return Open(ctx, g.store, full, g.opts)
}

// OpenArray opens a member array.
func (g *Group) OpenArray(ctx context.Context, name string) (*Array, error) {
	n, err := g.Member(ctx, name)
	if err != nil {
		return nil, err
	}
	a, ok := n.(*Array)
	if !ok {
		return nil, fmt.Errorf("zarr: %q is a group, not an array", name)
	}
	return a, nil
}

// OpenGroup opens a member group.
func (g *Group) OpenGroup(ctx context.Context, name string) (*Group, error) {
	n, err := g.Member(ctx, name)
	if err != nil {
		return nil, err
	}
	sub, ok := n.(*Group)
	if !ok {
		return nil, fmt.Errorf("zarr: %q is an array, not a group", name)
	}
	return sub, nil
}

// MemberNames returns the names of the group's children, sorted. Without
// consolidated metadata the store must be listable; entries without Zarr
// metadata (implicit groups) are skipped, as in zarr-python.
func (g *Group) MemberNames(ctx context.Context) ([]string, error) {
	if g.consolidated != nil {
		var out []string
		for k, e := range g.consolidated {
			if !strings.Contains(k, "/") && e.doc != nil {
				out = append(out, k)
			}
		}
		slices.Sort(out)
		return out, nil
	}
	ls, ok := g.store.(ListableStore)
	if !ok {
		return nil, fmt.Errorf("zarr: listing members needs a listable store or consolidated metadata")
	}
	prefix := g.path
	if prefix != "" {
		prefix += "/"
	}
	names, err := ls.ListDir(ctx, prefix)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, n := range names {
		switch n {
		case "zarr.json", ".zattrs", ".zgroup", ".zarray", ".zmetadata":
			continue
		}
		if _, err := readNode(ctx, g.store, joinKey(g.path, n)); err == nil {
			out = append(out, n)
		} else if !errors.Is(err, ErrNotFound) {
			return nil, err
		}
	}
	return out, nil
}

// Members opens the group's children.
func (g *Group) Members(ctx context.Context) ([]Node, error) {
	names, err := g.MemberNames(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]Node, 0, len(names))
	for _, n := range names {
		node, err := g.Member(ctx, n)
		if err != nil {
			return nil, err
		}
		out = append(out, node)
	}
	return out, nil
}

// Walk visits every descendant of the group (depth first, sorted by name)
// with its path relative to the group.
func (g *Group) Walk(ctx context.Context, fn func(rel string, n Node) error) error {
	var walk func(sub *Group, prefix string) error
	walk = func(sub *Group, prefix string) error {
		nodes, err := sub.Members(ctx)
		if err != nil {
			return err
		}
		for _, n := range nodes {
			rel := joinKey(prefix, path.Base(n.Path()))
			if err := fn(rel, n); err != nil {
				return err
			}
			if child, ok := n.(*Group); ok {
				if err := walk(child, rel); err != nil {
					return err
				}
			}
		}
		return nil
	}
	return walk(g, "")
}

// ---------------------------------------------------------------------------
// Creating nodes

// GroupSpec describes a new group.
type GroupSpec struct {
	// ZarrFormat is 3 (default) or 2.
	ZarrFormat int
	Attributes map[string]any
}

func groupDocs(format int, attrs any) (map[string][]byte, error) {
	if format == 2 {
		zg, _ := marshalPython(jsonObject{{"zarr_format", 2}}, 2)
		za, err := marshalPython(attrs, 2)
		if err != nil {
			return nil, err
		}
		return map[string][]byte{".zgroup": zg, ".zattrs": za}, nil
	}
	doc, err := marshalPython(jsonObject{{"attributes", attrs}, {"zarr_format", 3}, {"node_type", "group"}}, 2)
	if err != nil {
		return nil, err
	}
	return map[string][]byte{"zarr.json": doc}, nil
}

// CreateGroup creates a group at path (and any missing parent groups).
func CreateGroup(ctx context.Context, store WritableStore, p string, spec GroupSpec, opts *CreateOptions) (*Group, error) {
	p, err := normalizePath(p)
	if err != nil {
		return nil, err
	}
	format := spec.ZarrFormat
	if format == 0 {
		format = 3
	}
	if err := prepareNode(ctx, store, p, format, opts); err != nil {
		return nil, err
	}
	docs, err := groupDocs(format, attrsJSON(spec.Attributes))
	if err != nil {
		return nil, err
	}
	if err := setDocs(ctx, store, p, docs); err != nil {
		return nil, err
	}
	var o *Options
	if opts != nil {
		o = &opts.Options
	}
	return OpenGroup(ctx, store, p, o)
}

// CreateArray creates an array at path (and any missing parent groups).
func CreateArray(ctx context.Context, store WritableStore, p string, spec ArraySpec, opts *CreateOptions) (*Array, error) {
	p, err := normalizePath(p)
	if err != nil {
		return nil, err
	}
	m, err := spec.build()
	if err != nil {
		return nil, err
	}
	if err := prepareNode(ctx, store, p, m.ZarrFormat, opts); err != nil {
		return nil, err
	}
	if err := writeMetadata(ctx, store, p, m); err != nil {
		return nil, err
	}
	var o *Options
	if opts != nil {
		o = &opts.Options
	}
	return OpenArray(ctx, store, p, o)
}

// CreateGroup creates a member group.
func (g *Group) CreateGroup(ctx context.Context, name string, spec GroupSpec, opts *CreateOptions) (*Group, error) {
	ws, ok := g.store.(WritableStore)
	if !ok {
		return nil, ErrReadOnly
	}
	if spec.ZarrFormat == 0 {
		spec.ZarrFormat = g.format
	}
	return CreateGroup(ctx, ws, joinKey(g.path, name), spec, opts)
}

// CreateArray creates a member array.
func (g *Group) CreateArray(ctx context.Context, name string, spec ArraySpec, opts *CreateOptions) (*Array, error) {
	ws, ok := g.store.(WritableStore)
	if !ok {
		return nil, ErrReadOnly
	}
	if spec.ZarrFormat == 0 {
		spec.ZarrFormat = g.format
	}
	return CreateArray(ctx, ws, joinKey(g.path, name), spec, opts)
}

// SetAttributes replaces the group's attributes.
func (g *Group) SetAttributes(ctx context.Context, attrs map[string]any) error {
	ws, ok := g.store.(WritableStore)
	if !ok {
		return ErrReadOnly
	}
	raw := attrsJSON(attrs)
	var docs map[string][]byte
	var err error
	if g.format == 3 {
		doc := slices.Clone(g.doc)
		doc.set("attributes", raw)
		b, err := marshalPython(doc, 2)
		if err != nil {
			return err
		}
		docs = map[string][]byte{"zarr.json": b}
		g.doc = doc
	} else if docs, err = groupDocs(2, raw); err != nil {
		return err
	}
	if err := setDocs(ctx, ws, g.path, docs); err != nil {
		return err
	}
	b, _ := marshalPython(raw, -1)
	g.attrs, _ = decodeAttributes(b)
	g.raw = raw
	return nil
}

// SetAttributes replaces the array's attributes (for arrays opened from a
// writable store).
func (a *Array) SetAttributes(ctx context.Context, attrs map[string]any) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	ws, ok := a.store.(WritableStore)
	if !ok {
		return fmt.Errorf("%w: updating attributes needs a writable store", ErrReadOnly)
	}
	m := *a.meta
	m.doc = slices.Clone(a.meta.doc)
	if err := m.setAttributes(attrs); err != nil {
		return err
	}
	if err := writeMetadata(ctx, ws, a.path, &m); err != nil {
		return err
	}
	*a.meta = m
	return nil
}

func writeMetadata(ctx context.Context, store WritableStore, p string, m *Metadata) error {
	docs, err := m.documents()
	if err != nil {
		return err
	}
	return setDocs(ctx, store, p, docs)
}

func setDocs(ctx context.Context, store WritableStore, p string, docs map[string][]byte) error {
	names := make([]string, 0, len(docs))
	for n := range docs {
		names = append(names, n)
	}
	slices.Sort(names)
	for _, n := range names {
		if err := store.Set(ctx, joinKey(p, n), docs[n]); err != nil {
			return err
		}
	}
	return nil
}

// prepareNode checks that p is free (or clears it when overwriting) and
// creates missing parent groups, as zarr-python does.
func prepareNode(ctx context.Context, store WritableStore, p string, format int, opts *CreateOptions) error {
	if _, err := readNode(ctx, store, p); err == nil {
		if opts == nil || !opts.Overwrite {
			return fmt.Errorf("%w: %q", ErrExists, p)
		}
		if err := deleteNode(ctx, store, p); err != nil {
			return err
		}
	} else if !errors.Is(err, ErrNotFound) {
		return err
	}
	if p == "" {
		return nil
	}
	parts := strings.Split(p, "/")
	for i := range parts {
		parent := strings.Join(parts[:i], "/")
		n, err := readNode(ctx, store, parent)
		if err == nil {
			if n.nodeType == "array" {
				return fmt.Errorf("zarr: cannot create %q inside array %q", p, parent)
			}
			continue
		}
		if !errors.Is(err, ErrNotFound) {
			return err
		}
		docs, err := groupDocs(format, jsonObject{})
		if err != nil {
			return err
		}
		if err := setDocs(ctx, store, parent, docs); err != nil {
			return err
		}
	}
	return nil
}

// deleteNode removes a node's metadata and, if the store can list, everything below it.
func deleteNode(ctx context.Context, store WritableStore, p string) error {
	for _, k := range []string{"zarr.json", ".zarray", ".zgroup", ".zattrs", ".zmetadata"} {
		if err := store.Delete(ctx, joinKey(p, k)); err != nil {
			return err
		}
	}
	ls, ok := store.(ListableStore)
	if !ok {
		return nil
	}
	prefix := p
	if prefix != "" {
		prefix += "/"
	}
	keys, err := ls.List(ctx, prefix)
	if err != nil {
		return err
	}
	for _, k := range keys {
		if err := store.Delete(ctx, k); err != nil {
			return err
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// Consolidated metadata

// ConsolidateMetadata gathers the metadata of every node below the group
// at path into the group's metadata, in zarr-python's formats: inline
// "consolidated_metadata" in zarr.json (Zarr v3) or .zmetadata (Zarr v2).
// The store must be listable. Later changes are not reflected until it is
// called again.
func ConsolidateMetadata(ctx context.Context, store WritableStore, p string) (*Group, error) {
	p, err := normalizePath(p)
	if err != nil {
		return nil, err
	}
	n, err := readNode(ctx, store, p)
	if err != nil {
		return nil, err
	}
	if n.nodeType != "group" {
		return nil, fmt.Errorf("zarr: %q is not a group", p)
	}
	// Open without consolidated metadata to discover the real members.
	if n.format == 3 {
		o, err := parseOrdered(n.doc)
		if err != nil {
			return nil, err
		}
		doc := o.(jsonObject)
		doc.del("consolidated_metadata")
		n.doc, _ = marshalPython(doc, 2)
	}
	root, err := groupFromDocs(ctx, store, p, n, nil)
	if err != nil {
		return nil, err
	}
	root.consolidated = nil
	type entry struct {
		rel  string
		node Node
	}
	var entries []entry
	if err := root.Walk(ctx, func(rel string, node Node) error {
		entries = append(entries, entry{rel, node})
		return nil
	}); err != nil {
		return nil, err
	}
	sort.SliceStable(entries, func(i, j int) bool {
		di, dj := strings.Count(entries[i].rel, "/"), strings.Count(entries[j].rel, "/")
		if di != dj {
			return di < dj
		}
		return strings.ToLower(entries[i].rel) < strings.ToLower(entries[j].rel)
	})
	emptyConsolidated := jsonObject{{"kind", "inline"}, {"must_understand", false}, {"metadata", jsonObject{}}}
	if root.format == 3 {
		meta := jsonObject{}
		for _, e := range entries {
			switch node := e.node.(type) {
			case *Array:
				meta = append(meta, jsonMember{e.rel, node.meta.doc})
			case *Group:
				meta = append(meta, jsonMember{e.rel, jsonObject{{"attributes", groupAttrs(node)}, {"zarr_format", 3}, {"consolidated_metadata", emptyConsolidated}, {"node_type", "group"}}})
			}
		}
		doc := jsonObject{{"attributes", groupAttrs(root)}, {"zarr_format", 3},
			{"consolidated_metadata", jsonObject{{"kind", "inline"}, {"must_understand", false}, {"metadata", meta}}},
			{"node_type", "group"}}
		b, err := marshalPython(doc, 2)
		if err != nil {
			return nil, err
		}
		if err := store.Set(ctx, joinKey(p, "zarr.json"), b); err != nil {
			return nil, err
		}
	} else {
		meta := jsonObject{{".zgroup", jsonObject{{"zarr_format", 2}}}, {".zattrs", groupAttrs(root)}}
		for _, e := range entries {
			switch node := e.node.(type) {
			case *Array:
				meta = append(meta, jsonMember{e.rel + "/.zattrs", arrayAttrs(node)}, jsonMember{e.rel + "/.zarray", node.meta.doc})
			case *Group:
				meta = append(meta, jsonMember{e.rel + "/.zattrs", groupAttrs(node)},
					jsonMember{e.rel + "/.zgroup", jsonObject{{"zarr_format", 2}, {"consolidated_metadata", jsonObject{{"metadata", jsonObject{}}, {"must_understand", false}, {"kind", "inline"}}}}})
			}
		}
		docs, err := groupDocs(2, groupAttrs(root))
		if err != nil {
			return nil, err
		}
		docs[".zmetadata"], err = marshalPython(jsonObject{{"metadata", meta}, {"zarr_consolidated_format", 1}}, -1)
		if err != nil {
			return nil, err
		}
		if err := setDocs(ctx, store, p, docs); err != nil {
			return nil, err
		}
	}
	return OpenGroup(ctx, store, p, nil)
}

func groupAttrs(g *Group) any {
	if g.raw != nil {
		return g.raw
	}
	return attrsJSON(g.attrs)
}

func arrayAttrs(a *Array) any {
	if a.meta.rawAttrs != nil {
		return a.meta.rawAttrs
	}
	return attrsJSON(a.meta.Attributes)
}
