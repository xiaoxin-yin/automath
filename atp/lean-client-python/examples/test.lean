import Mathlib.Data.Set.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Data.Complex.Exponential
import Mathlib.Analysis.SpecialFunctions.Log.Basic

variable {α : Type*}
variable (s t u : Set α)
open Set

example (h : s ⊆ t) : s ∩ u ⊆ t ∩ u := by
  rw [subset_def, inter_def, inter_def]
  rw [subset_def] at h
  simp only [mem_setOf]
  rintro x ⟨xs, xu⟩
  exact ⟨h _ xs, xu⟩
  
def main : IO Unit :=
  IO.println s!"Hello!"