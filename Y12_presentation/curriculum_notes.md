4.2 Assembly Language

Candidates should be able to: 
- Show understanding of the relationship between assembly language and machine code
- Describe the different stages of the assembly process for a two-pass assembler
- Trace a given simple assembly language program
- Show understanding that a set of instructions are grouped
- Show understanding of and be able to use different modes of addressing

Notes and guidance:
- Apply the two-pass assembler process to a given simple assembly language program
- Including the following groups:
• Data movement
• Input and output of data
• Arithmetic operations
• Unconditional and conditional instructions
• Compare instructions
- Including immediate, direct, indirect, indexed, relative

The following table is an example of an instruction set:

Opcode & Operand & Instruction Explanation
LDM #n Immediate addressing. Load the number n to ACC
LDD <address> Direct addressing. Load the contents of the location at the given address to ACC
LDI <address> Indirect addressing. The address to be used is at the given address.
Load the contents of this second address to ACC
LDX <address> Indexed addressing. Form the address from <address> + the contents of the index register. Copy the contents of this calculated address to ACC
LDR #n Immediate addressing. Load the number n to IX
MOV <register> Move the contents of the accumulator to the given register (IX)
STO <address> Store the contents of ACC at the given address
ADD <address> Add the contents of the given address to the ACC
ADD #n/Bn/&n Add the number n to the ACC
SUB <address> Subtract the contents of the given address from the ACC
SUB #n/Bn/&n Subtract the number n from the ACC
INC <register> Add 1 to the contents of the register (ACC or IX)
DEC <register> Subtract 1 from the contents of the register (ACC or IX)
JMP <address> Jump to the given address
CMP <address> Compare the contents of ACC with the contents of <address>
CMP #n Compare the contents of ACC with number n
CMI <address> Indirect addressing. The address to be used is at the given address. Compare the contents of ACC with the contents of this second address
JPE <address> Following a compare instruction, jump to <address> if the compare was True
JPN <address> Following a compare instruction, jump to <address> if the compare was False
IN Key in a character and store its ASCII value in ACC
OUT Output to the screen the character whose ASCII value is stored in ACC
END Return control to the operating system
All questions will assume there is only one general purpose register available (Accumulator)
ACC denotes Accumulator
IX denotes Index Register

4.3 Bit manipulation
Candidates should be able to:
Show understanding of and perform binary shifts
Logical, arithmetic and cyclic Left shift, right shift
Notes and guidance
Show understanding of how bit manipulation can be used to monitor/control a device
Carry out bit manipulation operations
Test and set a bit (using bit masking)

Example Instruction Set
Label Opcode Operand Explanation
AND #n / Bn / &n Bitwise AND operation of the contents of ACC with the operand
AND <address> Bitwise AND operation of the contents of ACC with the contents of <address>
XOR #n / Bn / &n Bitwise XOR operation of the contents of ACC with the operand
XOR <address> Bitwise XOR operation of the contents of ACC with the contents of <address>
OR #n / Bn / &n Bitwise OR operation of the contents of ACC with the operand
OR <address> Bitwise OR operation of the contents of ACC with the contents of <address>
LSL #n Bits in ACC are shifted logically n places to the left. Zeros are introduced on the right hand end
LSR #n Bits in ACC are shifted logically n places to the right. Zeros are introduced on the left hand end
<label>: <opcode> <operand> Labels an instruction
<label>: <data> Gives a symbolic address <label> to the memory location with contents <data>
All questions will assume there is only one general purpose register available (Accumulator)
ACC denotes Accumulator
IX denotes Index Register


<address> can be an absolute or symbolic address
# denotes a denary number, e.g. #123
B denotes a binary number, e.g. B01001010
& denotes a hexadecimal number, e.g. &4A

5.2 Language Translators

Candidates should be able to: 
Show understanding of the need for:
• assembler software for the translation of an assembly language program
• a compiler for the translation of a high-level language program
• an interpreter for translation and execution of a high-level language program
Explain the benefits and drawbacks of using either a compiler or interpreter and justify the use of each
Show awareness that high-level language programs may be partially compiled and partially interpreted, such as Java (console mode)
Describe features found in a typical Integrated Development Environment (IDE) Including:
• for coding, including context-sensitive prompts
• for initial error detection, including dynamic
syntax checks
• for presentation, including prettyprint, expand
and collapse code blocks
• for debugging, including single stepping,
breakpoints, i.e. variables, expressions, report
window
