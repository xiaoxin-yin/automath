import Lake
open Lake DSL

package my_project {
  -- Add more configuration here
}

lean_lib MyProject

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"@"master"