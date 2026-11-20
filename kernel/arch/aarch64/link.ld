/* Silica kernel, aarch64. Linked position-independent at the kernel base;
   the loader adds a random slide and applies the relative relocations. */
ENTRY(_start)
PHDRS {
  text PT_LOAD FLAGS(5);
  rodata PT_LOAD FLAGS(4);
  data PT_LOAD FLAGS(6);
  dynamic PT_DYNAMIC FLAGS(4);
}
SECTIONS {
  . = 0xFFFFFFFF80000000;
  __kernel_start = .;
  .text.fast : { *(.text.fast*) } :text
  .text : { *(.text.boot) *(.text .text.*) } :text
  . = ALIGN(4096);
  .rodata : { *(.rodata .rodata.*) } :rodata
  .dynsym : { *(.dynsym) } :rodata
  .gnu.hash : { *(.gnu.hash) } :rodata
  .hash : { *(.hash) } :rodata
  .dynstr : { *(.dynstr) } :rodata
  .rela.dyn : { *(.rela*) } :rodata
  .dynamic : { *(.dynamic) } :rodata :dynamic
  . = ALIGN(4096);
  .data : { *(.data .data.*) *(.got .got.*) } :data
  .bss : { *(.bss .bss.*) *(COMMON) } :data
  . = ALIGN(4096);
  __kernel_end = .;
  /DISCARD/ : { *(.eh_frame*) *(.note*) *(.comment) }
}
