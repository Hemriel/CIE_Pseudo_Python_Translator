# Extended Lesson-Prep Notes — Assembly Language, Bit Manipulation, Translators (CIE 9618 AS)

These notes are written as **teacher preparation** (not slides). They are structured to align with typical **Cambridge International AS Computer Science (9618)** assessment style: short, precise definitions; tracing/working; applying processes (e.g., two-pass assembly); and interpreting small fragments of code/data.

---

## 0) What CIE tends to assess (practical exam focus)

**Students are usually expected to be able to:**
- **Define and distinguish** key terms (assembly vs machine code; compiler vs interpreter; addressing modes; masking).
- **Trace** simple assembly programs using a given instruction set (ACC + IX) and memory table.
- **Apply the two-pass assembler idea** to a small program: build a symbol table, resolve forward references, and translate to machine code.
- **Use addressing modes** (immediate/direct/indirect/indexed/relative) correctly in explanations and traces.
- **Do small bit operations by hand** (AND/OR/XOR) and **explain** shifts (logical/arithmetic/cyclic) and masking in device-control contexts.

**Common marks are in method, not just the final answer:**
- Showing state changes (ACC/IX/memory/flags) step-by-step.
- Correctly interpreting operand prefixes (`#`, `B`, `&`).
- Using the correct *effective address* for each addressing mode.

---

## 1) Assembly language ↔ machine code (relationship)

### Key idea
- **Machine code** is the binary/hex representation of CPU instructions that hardware executes.
- **Assembly language** is a human-readable mnemonic form of those instructions.
- An **assembler** translates assembly into machine code.

### What to emphasize in class
- Assembly is **architecture/instruction-set specific** (mnemonics depend on the CPU design).
- Assembly is still **low-level**: explicit control of registers and memory.
- A single assembly instruction typically corresponds to **one machine instruction** (contrast with high-level languages).

### Quick check questions
- “Why is assembly language called a ‘low-level’ language?”
- “Is machine code portable between different CPUs? Why/why not?”

---

## 2) The given model instruction set (what students must internalize)

Your curriculum sheet defines a **model CPU**:
- One general-purpose register: **ACC**
- Index register: **IX**
- Instructions include: `LDM`, `LDD`, `LDI`, `LDX`, `LDR`, `MOV`, `STO`, arithmetic, compare + conditional jump, I/O, bitwise ops, shifts.

### Operand formats
- `#n` = denary literal (immediate)
- `Bn...` = binary literal
- `&n...` = hex literal
- `<address>` = absolute address or symbolic label

### “Groups of instructions” (explicitly teach as categories)
- **Data movement**: load/store/move (`LDM`, `LDD`, `LDI`, `LDX`, `LDR`, `MOV`, `STO`)
- **Arithmetic**: `ADD`, `SUB`, `INC`, `DEC`
- **Compare + branch**: `CMP`, `CMI`, then `JPE`/`JPN`
- **Control flow**: `JMP`, conditional jumps
- **I/O**: `IN`, `OUT`
- **Bitwise / shifts**: `AND`, `OR`, `XOR`, `LSL`, `LSR`

Teaching tip: students lose marks when they **don’t connect** “compare” with “conditional jump”. Treat `CMP`/`CMI` as setting a *boolean result/flag* used by `JPE`/`JPN`.

---

## 3) Addressing modes (must be precise)

Teach each mode using the language:
- “What is in the instruction?”
- “How do we find the **effective address**?”
- “What value ends up in ACC?”

### Immediate addressing
- Example: `LDM #10`
- Meaning: load the **literal 10** into ACC.

### Direct addressing
- Example: `LDD 200`
- Meaning: use address 200 directly; load **memory[200]** into ACC.

### Indirect addressing
- Example: `LDI 200`
- Meaning: address 200 holds another address.
  - Step 1: read pointer = memory[200]
  - Step 2: load memory[pointer] into ACC

### Indexed addressing
- Example: `LDX 200`
- Meaning: effective address = 200 + IX; load memory[200 + IX] into ACC.

### Relative addressing (conceptual)
Your sample instruction set uses jumps like `JMP <address>` and conditional jumps `JPE <address>`.
- **Relative addressing** typically means the operand is an **offset** from the current instruction address (e.g., PC + offset).
- In exam questions, they may describe “relative” in terms of “jump forward/back by N instructions/bytes”.

If your taught model doesn’t explicitly show PC-relative encoding, keep it conceptual:
- relative = offset-based; direct = absolute address.

### Micro-drill (5 minutes)
Give a memory table and ask students to compute the **effective address** for:
- one direct
- one indirect
- one indexed

Then check that they can distinguish:
- indirect (pointer in memory)
- indexed (base + index)

---

## 4) Tracing a simple assembly program (how to teach it)

### The method students should always use
Provide (or require) a table with columns:
- Step / Line
- Instruction
- ACC
- IX
- Memory changes (address/value)
- Compare result/flag (if used)
- Output (if any)

### Typical pitfalls
- Confusing `#n` (literal) with `<address>` (memory reference).
- Forgetting that `STO <address>` writes ACC **into memory**.
- Treating `LDI` as “load immediate” (it is **load indirect** here).
- Not updating IX after `LDR #n` or after `MOV IX` (depending on instruction).

### Suggested mini-program patterns
- Swap two memory locations
- Sum an array with indexed addressing
- Read characters until sentinel (I/O + compare + conditional jump)

---

## 5) Two-pass assembler (stages and what each pass does)

### Big picture
A **two-pass assembler** is used when:
- we allow **labels** (symbolic addresses), and
- we allow **forward references** (jumping to labels defined later).

Pass 1 builds knowledge; Pass 2 generates final machine code.

### Pass 1 (symbol table + address assignment)
Students should be able to describe:
- The assembler scans the program line-by-line.
- Maintains a **location counter** (address of each instruction/data).
- When it sees a **label**, it records in a **symbol table**:
  - label name → address
- It can also note pseudo-ops/data directives if your question includes them.

Output of Pass 1:
- Symbol table
- Annotated intermediate form (often: each line with its resolved address)

### Pass 2 (translation + resolution)
- Re-scan program.
- Translate mnemonics + addressing modes into **opcodes**.
- Replace label operands with **numeric addresses** using the symbol table.
- Output the machine code.

### Forward references
- Example: `JMP END_LABEL` appears before `END_LABEL:` is defined.
- Pass 1 records `END_LABEL` when it eventually appears.
- Pass 2 can then correctly encode the earlier jump.

### How CIE questions typically look
- “Complete the symbol table” (labels + addresses).
- “Show the machine code output for these lines” (often in hex/binary).
- “Explain why two passes are needed” (forward references + labels).

---

## 6) Bit manipulation (what to teach + how it’s assessed)

### 6.1 Bitwise operations (AND/OR/XOR)
Use truth tables + byte examples.
- `AND` is used for **masking/clearing** bits.
- `OR` is used for **setting** bits.
- `XOR` is used for **toggling** bits.

Suggested quick examples (8-bit):
- Clear bit 3: `ACC = ACC AND 11110111`
- Set bit 0: `ACC = ACC OR 00000001`
- Toggle bit 6: `ACC = ACC XOR 01000000`

Relate directly to your instruction set:
- `AND #n / Bn / &n`
- `OR #n / Bn / &n`
- `XOR #n / Bn / &n`

### 6.2 Shifts (logical, arithmetic, cyclic)
Students need both:
- **mechanics** (what happens to bits), and
- **meaning** (why it’s used).

**Logical shift left (LSL)**
- Bits move left; zeros come in on the right.
- Often corresponds to multiply by 2 (for unsigned) if no overflow.

**Logical shift right (LSR)**
- Bits move right; zeros come in on the left.
- Often corresponds to divide by 2 (for unsigned) ignoring remainder.

**Arithmetic shifts**
- Preserve sign bit for signed numbers.
- Right arithmetic shift repeats the leftmost bit.

**Cyclic/rotate shifts**
- Bits “wrap around” from one end to the other.
- Used in some low-level algorithms/crypto and bitfield operations.

Your model ISA explicitly provides `LSL #n` and `LSR #n` (logical). If you want to mention arithmetic/cyclic, do it conceptually unless your questions add those instructions.

### 6.3 Bit masking for device monitoring/control
Teach a **register-as-bitfield** story:
- Each bit represents a sensor/flag/status.
  - bit 0: power on
  - bit 1: error
  - bit 2: motor running
  - …

**Test a bit** (mask then compare)
- `ACC = ACC AND mask`
- If result equals mask → bit(s) set.

**Set a bit**
- `ACC = ACC OR mask`

**Clear a bit**
- `ACC = ACC AND (NOT mask)` (if NOT exists) or AND with a constant that has 0 in that bit position.

Even if your ISA doesn’t define `NOT`, exam questions still often test “masking idea”. You can teach clearing by providing the precomputed mask (e.g., `11110111`).

---

## 7) Practice tasks you can reuse (low-prep)

### Task A — Addressing mode identification
Give 6 instructions (mix of `LDM`, `LDD`, `LDI`, `LDX`) and ask:
- mode, effective address (if any), and value loaded.

### Task B — Trace with a state table
Provide:
- initial ACC/IX
- a small memory table
- a 8–12 line program (one compare + conditional jump)

Students fill the state table.

### Task C — Two-pass assembler mini-question
Provide a program with 3–5 labels including one forward reference.
Ask:
1) Fill symbol table
2) Replace labels with addresses
3) (Optional) write pseudo machine-code format like `OP operand` if real encoding isn’t specified.

### Task D — Masking for device flags
Given a status byte in ACC, students:
- determine whether a named flag is set (using AND)
- set/clear a named flag

---

## 8) Translators + IDE (supporting content from your notes)

### Assembler
- Translates **assembly → machine code**.
- Needed because CPU runs machine code, not mnemonics/labels.

### Compiler vs interpreter
- **Compiler** translates whole program before execution.
  - Pros: faster execution; many errors found before running.
  - Cons: compile step; platform-specific binaries.
- **Interpreter** translates and executes line-by-line / statement-by-statement.
  - Pros: easier debugging/rapid iteration; portable source.
  - Cons: slower; many errors appear at runtime.

Many modern systems are mixed/hybrid (e.g., Java: compile to bytecode + interpreted/JIT).

### IDE features (examples to point out)
- Context-sensitive prompts/autocomplete
- Syntax highlighting + linting
- Dynamic checks / live error detection
- Pretty-print / formatting / folding (collapse blocks)
- Debugger: breakpoints, single-step, watch expressions/variables

---

## 9) Resources (references + videos)

### Official CIE / Cambridge International
- 9618 syllabus landing page: https://www.cambridgeinternational.org/programmes-and-qualifications/cambridge-international-as-and-a-level-computer-science-9618/
- 2024–2025 syllabus PDF: https://www.cambridgeinternational.org/Images/636089-2024-2025-syllabus.pdf
- 2026 syllabus PDF: https://www.cambridgeinternational.org/Images/697372-2026-syllabus.pdf
- 2027–2029 syllabus PDF: https://www.cambridgeinternational.org/Images/721397-2027-2029-syllabus.pdf

### Stable general references (good for teacher refreshers)
- Two-pass assembler overview (concept + why two passes): https://en.wikipedia.org/wiki/Two-pass_assembler
- Addressing modes (definitions + examples): https://en.wikipedia.org/wiki/Addressing_mode
- Bitwise operations (AND/OR/XOR, masking, shifts): https://en.wikipedia.org/wiki/Bitwise_operation
- Compiler / interpreter / IDE (translator context):
  - Compiler: https://en.wikipedia.org/wiki/Compiler
  - Interpreter: https://en.wikipedia.org/wiki/Interpreter_(computing)
  - IDE: https://en.wikipedia.org/wiki/Integrated_development_environment

### Short YouTube links (assembly-focused)
Because YouTube URLs and availability change, these are **robust search links** that reliably surface short explainer videos:
- “assembly language in 10 minutes” search: https://www.youtube.com/results?search_query=assembly+language+in+10+minutes
- “two pass assembler” search: https://www.youtube.com/results?search_query=two+pass+assembler
- “addressing modes immediate direct indirect indexed relative” search: https://www.youtube.com/results?search_query=addressing+modes+immediate+direct+indirect+indexed+relative
- “bit masking explained” search: https://www.youtube.com/results?search_query=bit+masking+explained
- “logical vs arithmetic shift” search: https://www.youtube.com/results?search_query=logical+shift+vs+arithmetic+shift

If you want, tell me which CPU context you prefer (generic vs ARM vs x86 vs 6502), and I can replace these with **specific** short videos that match your teaching narrative.

---

## 10) Glossary (student-facing definitions you can reuse)
- **Assembly language**: mnemonic form of machine instructions for a specific CPU.
- **Machine code**: binary/hex instructions executed directly by the CPU.
- **Assembler**: translates assembly language to machine code.
- **Two-pass assembler**: assembler that uses one pass to build a symbol table and a second pass to generate final machine code.
- **Symbol table**: mapping of labels/symbols to addresses.
- **Immediate addressing**: operand is a literal value.
- **Direct addressing**: operand is an address of the data.
- **Indirect addressing**: operand points to a memory location holding the real address.
- **Indexed addressing**: effective address is base address + index register.
- **Relative addressing**: effective address is based on an offset from the current instruction location.
- **Masking**: using AND/OR/XOR with a mask value to test/set/clear bits.
