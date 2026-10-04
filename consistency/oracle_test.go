package consistency

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"
)

// oracle is a running consistency/oracle.py.
type oracle struct {
	mu  sync.Mutex
	cmd *exec.Cmd
	in  io.WriteCloser
	out *bufio.Reader
}

var (
	oracleOnce sync.Once
	theOracle  *oracle
	oracleErr  error
)

// pythonOracle returns the shared oracle, skipping the test if
// ZARR_PYTHON does not name an interpreter.
func pythonOracle(t testing.TB) *oracle {
	t.Helper()
	py := os.Getenv("ZARR_PYTHON")
	if py == "" {
		t.Skip("set ZARR_PYTHON to a Python with zarr-python (see consistency/README.md) to run")
	}
	oracleOnce.Do(func() {
		if !filepath.IsAbs(py) {
			if abs, err := filepath.Abs(filepath.Join("..", py)); err == nil {
				if _, err := os.Stat(abs); err == nil {
					py = abs
				}
			}
		}
		cmd := exec.Command(py, "oracle.py")
		cmd.Stderr = os.Stderr
		in, err := cmd.StdinPipe()
		if err != nil {
			oracleErr = err
			return
		}
		out, err := cmd.StdoutPipe()
		if err != nil {
			oracleErr = err
			return
		}
		if err := cmd.Start(); err != nil {
			oracleErr = err
			return
		}
		theOracle = &oracle{cmd: cmd, in: in, out: bufio.NewReaderSize(out, 1<<20)}
	})
	if oracleErr != nil {
		t.Fatalf("starting the oracle: %v", oracleErr)
	}
	return theOracle
}

// pyError is an exception raised in the oracle.
type pyError struct {
	Message string `json:"error"`
	Type    string `json:"type"`
	Trace   string `json:"trace"`
}

func (e *pyError) Error() string { return "python: " + e.Message }

// call runs one operation; result receives the "ok" value.
func (o *oracle) call(op string, req map[string]any, result any) error {
	o.mu.Lock()
	defer o.mu.Unlock()
	if req == nil {
		req = map[string]any{}
	}
	req["op"] = op
	line, err := json.Marshal(req)
	if err != nil {
		return err
	}
	if _, err := o.in.Write(append(line, '\n')); err != nil {
		return err
	}
	resp, err := o.out.ReadBytes('\n')
	if err != nil {
		return fmt.Errorf("oracle: %w", err)
	}
	var r struct {
		OK json.RawMessage `json:"ok"`
		pyError
	}
	if err := json.Unmarshal(resp, &r); err != nil {
		return fmt.Errorf("oracle: bad response %q: %w", resp, err)
	}
	if r.Message != "" {
		return &r.pyError
	}
	if result != nil {
		return json.Unmarshal(r.OK, result)
	}
	return nil
}

func TestMain(m *testing.M) {
	code := m.Run()
	if len(reports) > 0 {
		var ops, cmp, bytesEq, values, errs int
		for _, r := range reports {
			ops += r.Ops
			cmp += r.Comparisons
			bytesEq += r.ByteChunks
			values += r.ValueChunks
			errs += r.Errors
		}
		fmt.Printf("consistency: %d array cases, %d operations, %d comparisons (%d chunks byte-identical, %d chunks decoded and compared), %d operations both refused\n",
			len(reports), ops, cmp, bytesEq, values, errs)
	}
	if theOracle != nil {
		theOracle.call("exit", nil, nil)
		theOracle.in.Close()
		theOracle.cmd.Wait()
	}
	os.Exit(code)
}

// TestOracleVersions records the versions the comparison ran against.
func TestOracleVersions(t *testing.T) {
	o := pythonOracle(t)
	var v map[string]any
	if err := o.call("versions", nil, &v); err != nil {
		t.Fatal(err)
	}
	t.Logf("comparing against %v", v)
	if v["zarr"] != "3.4.0" || v["numcodecs"] != "0.17.0" {
		t.Logf("warning: the committed expectations were made with zarr 3.4.0 / numcodecs 0.17.0")
	}
}
