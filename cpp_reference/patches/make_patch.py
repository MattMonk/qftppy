"""Regenerate tensorindex-permute.patch against the pinned qft++ commit.

Usage (from the repo root): pixi run -e dev python cpp_reference/patches/make_patch.py
Requires git on PATH and network access. (Use the dev environment: the cpp environment cannot be
built until this patch exists.)
"""

import pathlib
import subprocess
import tempfile

REPO = "https://github.com/jdalseno/qft.git"
COMMIT = "df048f3a8c33ffb8984671fb057acd7dcd345a61"
OUT = pathlib.Path(__file__).with_name("tensorindex-permute.patch")

NEW_PERMUTE = """TensorIndex& TensorIndex::Permute() {

  // Patched for qftppy tests: the original odometer loop never terminates for
  // rank 4, because the 2-bit index fields cannot exceed rank - 1 = 3. Step
  // through permutations of (0,...,rank-1) in the same lexicographic order with
  // std::next_permutation instead; _index = -1 marks the end. Index fields are
  // 2 bits wide, so permutations only exist for rank <= 4.
  if(_rank < 1 || _index < 0) return *this;
  if(_rank > 4){
    _index = -1;
    return (*this);
  }
  int d[4];
  bool distinct = true;
  for(unsigned int i = 0; i < _rank; ++i){
    d[i] = (*this)[i];
    for(unsigned int j = 0; j < i; ++j) if(d[i] == d[j]) distinct = false;
  }
  if(!distinct){ // not yet a permutation: start from the identity
    for(unsigned int i = 0; i < _rank; ++i) d[i] = i;
  }
  else if(!std::next_permutation(d, d + _rank)){
    _index = -1;
    return (*this);
  }
  _index = 0;
  for(unsigned int i = 0; i < _rank; ++i) this->SetIndex(i, d[i]);
  return (*this);
}
"""


def main():
    with tempfile.TemporaryDirectory() as tmp:
        src = pathlib.Path(tmp) / "qft"
        subprocess.run(["git", "clone", "-q", REPO, str(src)], check=True)
        subprocess.run(["git", "-C", str(src), "checkout", "-q", COMMIT], check=True)

        cfile = src / "src/tensor/TensorIndex.C"
        text = cfile.read_text()
        start = text.index("TensorIndex& TensorIndex::Permute() {")
        end = text.index("//___", start)
        text = "#include <algorithm>\n" + text[:start] + NEW_PERMUTE + text[end:]
        cfile.write_text(text)

        hfile = src / "src/tensor/TensorIndex.h"
        header = hfile.read_text()
        anchor = "  bool PermIsValid() const {\n"
        assert header.count(anchor) == 1
        hfile.write_text(header.replace(anchor, anchor + "    if(_index < 0) return false;\n"))

        diff = subprocess.run(
            ["git", "-C", str(src), "diff", "--", "src/tensor/TensorIndex.C", "src/tensor/TensorIndex.h"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    OUT.write_text(diff)
    print(f"wrote {OUT} ({len(diff.splitlines())} lines)")


if __name__ == "__main__":
    main()
