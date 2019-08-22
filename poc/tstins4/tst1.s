
tst1:     file format elf64-x86-64


Disassembly of section .init:

00000000004009e8 <_init>:
  4009e8:	48 83 ec 08          	sub    rsp,0x8
  4009ec:	48 8b 05 05 36 20 00 	mov    rax,QWORD PTR [rip+0x203605]        # 603ff8 <_DYNAMIC+0x1f0>
  4009f3:	48 85 c0             	test   rax,rax
  4009f6:	74 05                	je     4009fd <_init+0x15>
  4009f8:	e8 73 01 00 00       	call   400b70 <std::ios_base::Init::~Init()@plt+0x10>
  4009fd:	48 83 c4 08          	add    rsp,0x8
  400a01:	c3                   	ret    

Disassembly of section .plt:

0000000000400a10 <cudaSetupArgument@plt-0x10>:
  400a10:	ff 35 f2 35 20 00    	push   QWORD PTR [rip+0x2035f2]        # 604008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400a16:	ff 25 f4 35 20 00    	jmp    QWORD PTR [rip+0x2035f4]        # 604010 <_GLOBAL_OFFSET_TABLE_+0x10>
  400a1c:	0f 1f 40 00          	nop    DWORD PTR [rax+0x0]

0000000000400a20 <cudaSetupArgument@plt>:
  400a20:	ff 25 f2 35 20 00    	jmp    QWORD PTR [rip+0x2035f2]        # 604018 <_GLOBAL_OFFSET_TABLE_+0x18>
  400a26:	68 00 00 00 00       	push   0x0
  400a2b:	e9 e0 ff ff ff       	jmp    400a10 <_init+0x28>

0000000000400a30 <cudaMemset@plt>:
  400a30:	ff 25 ea 35 20 00    	jmp    QWORD PTR [rip+0x2035ea]        # 604020 <_GLOBAL_OFFSET_TABLE_+0x20>
  400a36:	68 01 00 00 00       	push   0x1
  400a3b:	e9 d0 ff ff ff       	jmp    400a10 <_init+0x28>

0000000000400a40 <__cudaRegisterFunction@plt>:
  400a40:	ff 25 e2 35 20 00    	jmp    QWORD PTR [rip+0x2035e2]        # 604028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400a46:	68 02 00 00 00       	push   0x2
  400a4b:	e9 c0 ff ff ff       	jmp    400a10 <_init+0x28>

0000000000400a50 <__cudaInitModule@plt>:
  400a50:	ff 25 da 35 20 00    	jmp    QWORD PTR [rip+0x2035da]        # 604030 <_GLOBAL_OFFSET_TABLE_+0x30>
  400a56:	68 03 00 00 00       	push   0x3
  400a5b:	e9 b0 ff ff ff       	jmp    400a10 <_init+0x28>

0000000000400a60 <__cxa_atexit@plt>:
  400a60:	ff 25 d2 35 20 00    	jmp    QWORD PTR [rip+0x2035d2]        # 604038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400a66:	68 04 00 00 00       	push   0x4
  400a6b:	e9 a0 ff ff ff       	jmp    400a10 <_init+0x28>

0000000000400a70 <cudaConfigureCall@plt>:
  400a70:	ff 25 ca 35 20 00    	jmp    QWORD PTR [rip+0x2035ca]        # 604040 <_GLOBAL_OFFSET_TABLE_+0x40>
  400a76:	68 05 00 00 00       	push   0x5
  400a7b:	e9 90 ff ff ff       	jmp    400a10 <_init+0x28>

0000000000400a80 <__stack_chk_fail@plt>:
  400a80:	ff 25 c2 35 20 00    	jmp    QWORD PTR [rip+0x2035c2]        # 604048 <_GLOBAL_OFFSET_TABLE_+0x48>
  400a86:	68 06 00 00 00       	push   0x6
  400a8b:	e9 80 ff ff ff       	jmp    400a10 <_init+0x28>

0000000000400a90 <exit@plt>:
  400a90:	ff 25 ba 35 20 00    	jmp    QWORD PTR [rip+0x2035ba]        # 604050 <_GLOBAL_OFFSET_TABLE_+0x50>
  400a96:	68 07 00 00 00       	push   0x7
  400a9b:	e9 70 ff ff ff       	jmp    400a10 <_init+0x28>

0000000000400aa0 <malloc@plt>:
  400aa0:	ff 25 b2 35 20 00    	jmp    QWORD PTR [rip+0x2035b2]        # 604058 <_GLOBAL_OFFSET_TABLE_+0x58>
  400aa6:	68 08 00 00 00       	push   0x8
  400aab:	e9 60 ff ff ff       	jmp    400a10 <_init+0x28>

0000000000400ab0 <fprintf@plt>:
  400ab0:	ff 25 aa 35 20 00    	jmp    QWORD PTR [rip+0x2035aa]        # 604060 <_GLOBAL_OFFSET_TABLE_+0x60>
  400ab6:	68 09 00 00 00       	push   0x9
  400abb:	e9 50 ff ff ff       	jmp    400a10 <_init+0x28>

0000000000400ac0 <std::ios_base::Init::Init()@plt>:
  400ac0:	ff 25 a2 35 20 00    	jmp    QWORD PTR [rip+0x2035a2]        # 604068 <_GLOBAL_OFFSET_TABLE_+0x68>
  400ac6:	68 0a 00 00 00       	push   0xa
  400acb:	e9 40 ff ff ff       	jmp    400a10 <_init+0x28>

0000000000400ad0 <puts@plt>:
  400ad0:	ff 25 9a 35 20 00    	jmp    QWORD PTR [rip+0x20359a]        # 604070 <_GLOBAL_OFFSET_TABLE_+0x70>
  400ad6:	68 0b 00 00 00       	push   0xb
  400adb:	e9 30 ff ff ff       	jmp    400a10 <_init+0x28>

0000000000400ae0 <cudaDeviceSynchronize@plt>:
  400ae0:	ff 25 92 35 20 00    	jmp    QWORD PTR [rip+0x203592]        # 604078 <_GLOBAL_OFFSET_TABLE_+0x78>
  400ae6:	68 0c 00 00 00       	push   0xc
  400aeb:	e9 20 ff ff ff       	jmp    400a10 <_init+0x28>

0000000000400af0 <cudaLaunch@plt>:
  400af0:	ff 25 8a 35 20 00    	jmp    QWORD PTR [rip+0x20358a]        # 604080 <_GLOBAL_OFFSET_TABLE_+0x80>
  400af6:	68 0d 00 00 00       	push   0xd
  400afb:	e9 10 ff ff ff       	jmp    400a10 <_init+0x28>

0000000000400b00 <__cudaUnregisterFatBinary@plt>:
  400b00:	ff 25 82 35 20 00    	jmp    QWORD PTR [rip+0x203582]        # 604088 <_GLOBAL_OFFSET_TABLE_+0x88>
  400b06:	68 0e 00 00 00       	push   0xe
  400b0b:	e9 00 ff ff ff       	jmp    400a10 <_init+0x28>

0000000000400b10 <__cudaRegisterFatBinary@plt>:
  400b10:	ff 25 7a 35 20 00    	jmp    QWORD PTR [rip+0x20357a]        # 604090 <_GLOBAL_OFFSET_TABLE_+0x90>
  400b16:	68 0f 00 00 00       	push   0xf
  400b1b:	e9 f0 fe ff ff       	jmp    400a10 <_init+0x28>

0000000000400b20 <__libc_start_main@plt>:
  400b20:	ff 25 72 35 20 00    	jmp    QWORD PTR [rip+0x203572]        # 604098 <_GLOBAL_OFFSET_TABLE_+0x98>
  400b26:	68 10 00 00 00       	push   0x10
  400b2b:	e9 e0 fe ff ff       	jmp    400a10 <_init+0x28>

0000000000400b30 <cudaMalloc@plt>:
  400b30:	ff 25 6a 35 20 00    	jmp    QWORD PTR [rip+0x20356a]        # 6040a0 <_GLOBAL_OFFSET_TABLE_+0xa0>
  400b36:	68 11 00 00 00       	push   0x11
  400b3b:	e9 d0 fe ff ff       	jmp    400a10 <_init+0x28>

0000000000400b40 <cudaMemcpy@plt>:
  400b40:	ff 25 62 35 20 00    	jmp    QWORD PTR [rip+0x203562]        # 6040a8 <_GLOBAL_OFFSET_TABLE_+0xa8>
  400b46:	68 12 00 00 00       	push   0x12
  400b4b:	e9 c0 fe ff ff       	jmp    400a10 <_init+0x28>

0000000000400b50 <cudaDeviceReset@plt>:
  400b50:	ff 25 5a 35 20 00    	jmp    QWORD PTR [rip+0x20355a]        # 6040b0 <_GLOBAL_OFFSET_TABLE_+0xb0>
  400b56:	68 13 00 00 00       	push   0x13
  400b5b:	e9 b0 fe ff ff       	jmp    400a10 <_init+0x28>

0000000000400b60 <std::ios_base::Init::~Init()@plt>:
  400b60:	ff 25 52 35 20 00    	jmp    QWORD PTR [rip+0x203552]        # 6040b8 <_GLOBAL_OFFSET_TABLE_+0xb8>
  400b66:	68 14 00 00 00       	push   0x14
  400b6b:	e9 a0 fe ff ff       	jmp    400a10 <_init+0x28>

Disassembly of section .plt.got:

0000000000400b70 <.plt.got>:
  400b70:	ff 25 82 34 20 00    	jmp    QWORD PTR [rip+0x203482]        # 603ff8 <_DYNAMIC+0x1f0>
  400b76:	66 90                	xchg   ax,ax

Disassembly of section .text:

0000000000400b80 <_start>:
  400b80:	31 ed                	xor    ebp,ebp
  400b82:	49 89 d1             	mov    r9,rdx
  400b85:	5e                   	pop    rsi
  400b86:	48 89 e2             	mov    rdx,rsp
  400b89:	48 83 e4 f0          	and    rsp,0xfffffffffffffff0
  400b8d:	50                   	push   rax
  400b8e:	54                   	push   rsp
  400b8f:	49 c7 c0 d0 17 40 00 	mov    r8,0x4017d0
  400b96:	48 c7 c1 60 17 40 00 	mov    rcx,0x401760
  400b9d:	48 c7 c7 27 13 40 00 	mov    rdi,0x401327
  400ba4:	e8 77 ff ff ff       	call   400b20 <__libc_start_main@plt>
  400ba9:	f4                   	hlt    
  400baa:	66 0f 1f 44 00 00    	nop    WORD PTR [rax+rax*1+0x0]

0000000000400bb0 <deregister_tm_clones>:
  400bb0:	b8 d7 40 60 00       	mov    eax,0x6040d7
  400bb5:	55                   	push   rbp
  400bb6:	48 2d d0 40 60 00    	sub    rax,0x6040d0
  400bbc:	48 83 f8 0e          	cmp    rax,0xe
  400bc0:	48 89 e5             	mov    rbp,rsp
  400bc3:	76 1b                	jbe    400be0 <deregister_tm_clones+0x30>
  400bc5:	b8 00 00 00 00       	mov    eax,0x0
  400bca:	48 85 c0             	test   rax,rax
  400bcd:	74 11                	je     400be0 <deregister_tm_clones+0x30>
  400bcf:	5d                   	pop    rbp
  400bd0:	bf d0 40 60 00       	mov    edi,0x6040d0
  400bd5:	ff e0                	jmp    rax
  400bd7:	66 0f 1f 84 00 00 00 	nop    WORD PTR [rax+rax*1+0x0]
  400bde:	00 00 
  400be0:	5d                   	pop    rbp
  400be1:	c3                   	ret    
  400be2:	0f 1f 40 00          	nop    DWORD PTR [rax+0x0]
  400be6:	66 2e 0f 1f 84 00 00 	nop    WORD PTR cs:[rax+rax*1+0x0]
  400bed:	00 00 00 

0000000000400bf0 <register_tm_clones>:
  400bf0:	be d0 40 60 00       	mov    esi,0x6040d0
  400bf5:	55                   	push   rbp
  400bf6:	48 81 ee d0 40 60 00 	sub    rsi,0x6040d0
  400bfd:	48 c1 fe 03          	sar    rsi,0x3
  400c01:	48 89 e5             	mov    rbp,rsp
  400c04:	48 89 f0             	mov    rax,rsi
  400c07:	48 c1 e8 3f          	shr    rax,0x3f
  400c0b:	48 01 c6             	add    rsi,rax
  400c0e:	48 d1 fe             	sar    rsi,1
  400c11:	74 15                	je     400c28 <register_tm_clones+0x38>
  400c13:	b8 00 00 00 00       	mov    eax,0x0
  400c18:	48 85 c0             	test   rax,rax
  400c1b:	74 0b                	je     400c28 <register_tm_clones+0x38>
  400c1d:	5d                   	pop    rbp
  400c1e:	bf d0 40 60 00       	mov    edi,0x6040d0
  400c23:	ff e0                	jmp    rax
  400c25:	0f 1f 00             	nop    DWORD PTR [rax]
  400c28:	5d                   	pop    rbp
  400c29:	c3                   	ret    
  400c2a:	66 0f 1f 44 00 00    	nop    WORD PTR [rax+rax*1+0x0]

0000000000400c30 <__do_global_dtors_aux>:
  400c30:	80 3d b1 34 20 00 00 	cmp    BYTE PTR [rip+0x2034b1],0x0        # 6040e8 <completed.7307>
  400c37:	75 11                	jne    400c4a <__do_global_dtors_aux+0x1a>
  400c39:	55                   	push   rbp
  400c3a:	48 89 e5             	mov    rbp,rsp
  400c3d:	e8 6e ff ff ff       	call   400bb0 <deregister_tm_clones>
  400c42:	5d                   	pop    rbp
  400c43:	c6 05 9e 34 20 00 01 	mov    BYTE PTR [rip+0x20349e],0x1        # 6040e8 <completed.7307>
  400c4a:	f3 c3                	repz ret 
  400c4c:	0f 1f 40 00          	nop    DWORD PTR [rax+0x0]

0000000000400c50 <frame_dummy>:
  400c50:	bf 00 3e 60 00       	mov    edi,0x603e00
  400c55:	48 83 3f 00          	cmp    QWORD PTR [rdi],0x0
  400c59:	75 05                	jne    400c60 <frame_dummy+0x10>
  400c5b:	eb 93                	jmp    400bf0 <register_tm_clones>
  400c5d:	0f 1f 00             	nop    DWORD PTR [rax]
  400c60:	b8 00 00 00 00       	mov    eax,0x0
  400c65:	48 85 c0             	test   rax,rax
  400c68:	74 f1                	je     400c5b <frame_dummy+0xb>
  400c6a:	55                   	push   rbp
  400c6b:	48 89 e5             	mov    rbp,rsp
  400c6e:	ff d0                	call   rax
  400c70:	5d                   	pop    rbp
  400c71:	e9 7a ff ff ff       	jmp    400bf0 <register_tm_clones>

0000000000400c76 <__cudaUnregisterBinaryUtil>:
  400c76:	55                   	push   rbp
  400c77:	48 89 e5             	mov    rbp,rsp
  400c7a:	48 8b 05 6f 34 20 00 	mov    rax,QWORD PTR [rip+0x20346f]        # 6040f0 <__cudaFatCubinHandle>
  400c81:	48 89 c7             	mov    rdi,rax
  400c84:	e8 77 fe ff ff       	call   400b00 <__cudaUnregisterFatBinary@plt>
  400c89:	5d                   	pop    rbp
  400c8a:	c3                   	ret    

0000000000400c8b <__nv_save_fatbinhandle_for_managed_rt(void**)>:
  400c8b:	55                   	push   rbp
  400c8c:	48 89 e5             	mov    rbp,rsp
  400c8f:	48 89 7d f8          	mov    QWORD PTR [rbp-0x8],rdi
  400c93:	48 8b 45 f8          	mov    rax,QWORD PTR [rbp-0x8]
  400c97:	48 89 05 6a 34 20 00 	mov    QWORD PTR [rip+0x20346a],rax        # 604108 <__nv_fatbinhandle_for_managed_rt>
  400c9e:	5d                   	pop    rbp
  400c9f:	c3                   	ret    

0000000000400ca0 <_cudaGetErrorEnum(cudaError)>:
  400ca0:	55                   	push   rbp
  400ca1:	48 89 e5             	mov    rbp,rsp
  400ca4:	89 7d fc             	mov    DWORD PTR [rbp-0x4],edi
  400ca7:	8b 45 fc             	mov    eax,DWORD PTR [rbp-0x4]
  400caa:	83 f8 27             	cmp    eax,0x27
  400cad:	0f 84 09 05 00 00    	je     4011bc <_cudaGetErrorEnum(cudaError)+0x51c>
  400cb3:	83 f8 27             	cmp    eax,0x27
  400cb6:	0f 8f b7 01 00 00    	jg     400e73 <_cudaGetErrorEnum(cudaError)+0x1d3>
  400cbc:	83 f8 13             	cmp    eax,0x13
  400cbf:	0f 84 2f 04 00 00    	je     4010f4 <_cudaGetErrorEnum(cudaError)+0x454>
  400cc5:	83 f8 13             	cmp    eax,0x13
  400cc8:	0f 8f d7 00 00 00    	jg     400da5 <_cudaGetErrorEnum(cudaError)+0x105>
  400cce:	83 f8 09             	cmp    eax,0x9
  400cd1:	0f 84 b9 03 00 00    	je     401090 <_cudaGetErrorEnum(cudaError)+0x3f0>
  400cd7:	83 f8 09             	cmp    eax,0x9
  400cda:	7f 69                	jg     400d45 <_cudaGetErrorEnum(cudaError)+0xa5>
  400cdc:	83 f8 04             	cmp    eax,0x4
  400cdf:	0f 84 79 03 00 00    	je     40105e <_cudaGetErrorEnum(cudaError)+0x3be>
  400ce5:	83 f8 04             	cmp    eax,0x4
  400ce8:	7f 32                	jg     400d1c <_cudaGetErrorEnum(cudaError)+0x7c>
  400cea:	83 f8 01             	cmp    eax,0x1
  400ced:	0f 84 4d 03 00 00    	je     401040 <_cudaGetErrorEnum(cudaError)+0x3a0>
  400cf3:	83 f8 01             	cmp    eax,0x1
  400cf6:	7f 0d                	jg     400d05 <_cudaGetErrorEnum(cudaError)+0x65>
  400cf8:	85 c0                	test   eax,eax
  400cfa:	0f 84 36 03 00 00    	je     401036 <_cudaGetErrorEnum(cudaError)+0x396>
  400d00:	e9 1b 06 00 00       	jmp    401320 <_cudaGetErrorEnum(cudaError)+0x680>
  400d05:	83 f8 02             	cmp    eax,0x2
  400d08:	0f 84 3c 03 00 00    	je     40104a <_cudaGetErrorEnum(cudaError)+0x3aa>
  400d0e:	83 f8 03             	cmp    eax,0x3
  400d11:	0f 84 3d 03 00 00    	je     401054 <_cudaGetErrorEnum(cudaError)+0x3b4>
  400d17:	e9 04 06 00 00       	jmp    401320 <_cudaGetErrorEnum(cudaError)+0x680>
  400d1c:	83 f8 06             	cmp    eax,0x6
  400d1f:	0f 84 4d 03 00 00    	je     401072 <_cudaGetErrorEnum(cudaError)+0x3d2>
  400d25:	83 f8 06             	cmp    eax,0x6
  400d28:	0f 8c 3a 03 00 00    	jl     401068 <_cudaGetErrorEnum(cudaError)+0x3c8>
  400d2e:	83 f8 07             	cmp    eax,0x7
  400d31:	0f 84 45 03 00 00    	je     40107c <_cudaGetErrorEnum(cudaError)+0x3dc>
  400d37:	83 f8 08             	cmp    eax,0x8
  400d3a:	0f 84 46 03 00 00    	je     401086 <_cudaGetErrorEnum(cudaError)+0x3e6>
  400d40:	e9 db 05 00 00       	jmp    401320 <_cudaGetErrorEnum(cudaError)+0x680>
  400d45:	83 f8 0e             	cmp    eax,0xe
  400d48:	0f 84 74 03 00 00    	je     4010c2 <_cudaGetErrorEnum(cudaError)+0x422>
  400d4e:	83 f8 0e             	cmp    eax,0xe
  400d51:	7f 29                	jg     400d7c <_cudaGetErrorEnum(cudaError)+0xdc>
  400d53:	83 f8 0b             	cmp    eax,0xb
  400d56:	0f 84 48 03 00 00    	je     4010a4 <_cudaGetErrorEnum(cudaError)+0x404>
  400d5c:	83 f8 0b             	cmp    eax,0xb
  400d5f:	0f 8c 35 03 00 00    	jl     40109a <_cudaGetErrorEnum(cudaError)+0x3fa>
  400d65:	83 f8 0c             	cmp    eax,0xc
  400d68:	0f 84 40 03 00 00    	je     4010ae <_cudaGetErrorEnum(cudaError)+0x40e>
  400d6e:	83 f8 0d             	cmp    eax,0xd
  400d71:	0f 84 41 03 00 00    	je     4010b8 <_cudaGetErrorEnum(cudaError)+0x418>
  400d77:	e9 a4 05 00 00       	jmp    401320 <_cudaGetErrorEnum(cudaError)+0x680>
  400d7c:	83 f8 10             	cmp    eax,0x10
  400d7f:	0f 84 51 03 00 00    	je     4010d6 <_cudaGetErrorEnum(cudaError)+0x436>
  400d85:	83 f8 10             	cmp    eax,0x10
  400d88:	0f 8c 3e 03 00 00    	jl     4010cc <_cudaGetErrorEnum(cudaError)+0x42c>
  400d8e:	83 f8 11             	cmp    eax,0x11
  400d91:	0f 84 49 03 00 00    	je     4010e0 <_cudaGetErrorEnum(cudaError)+0x440>
  400d97:	83 f8 12             	cmp    eax,0x12
  400d9a:	0f 84 4a 03 00 00    	je     4010ea <_cudaGetErrorEnum(cudaError)+0x44a>
  400da0:	e9 7b 05 00 00       	jmp    401320 <_cudaGetErrorEnum(cudaError)+0x680>
  400da5:	83 f8 1d             	cmp    eax,0x1d
  400da8:	0f 84 aa 03 00 00    	je     401158 <_cudaGetErrorEnum(cudaError)+0x4b8>
  400dae:	83 f8 1d             	cmp    eax,0x1d
  400db1:	7f 60                	jg     400e13 <_cudaGetErrorEnum(cudaError)+0x173>
  400db3:	83 f8 18             	cmp    eax,0x18
  400db6:	0f 84 6a 03 00 00    	je     401126 <_cudaGetErrorEnum(cudaError)+0x486>
  400dbc:	83 f8 18             	cmp    eax,0x18
  400dbf:	7f 29                	jg     400dea <_cudaGetErrorEnum(cudaError)+0x14a>
  400dc1:	83 f8 15             	cmp    eax,0x15
  400dc4:	0f 84 3e 03 00 00    	je     401108 <_cudaGetErrorEnum(cudaError)+0x468>
  400dca:	83 f8 15             	cmp    eax,0x15
  400dcd:	0f 8c 2b 03 00 00    	jl     4010fe <_cudaGetErrorEnum(cudaError)+0x45e>
  400dd3:	83 f8 16             	cmp    eax,0x16
  400dd6:	0f 84 36 03 00 00    	je     401112 <_cudaGetErrorEnum(cudaError)+0x472>
  400ddc:	83 f8 17             	cmp    eax,0x17
  400ddf:	0f 84 37 03 00 00    	je     40111c <_cudaGetErrorEnum(cudaError)+0x47c>
  400de5:	e9 36 05 00 00       	jmp    401320 <_cudaGetErrorEnum(cudaError)+0x680>
  400dea:	83 f8 1a             	cmp    eax,0x1a
  400ded:	0f 84 47 03 00 00    	je     40113a <_cudaGetErrorEnum(cudaError)+0x49a>
  400df3:	83 f8 1a             	cmp    eax,0x1a
  400df6:	0f 8c 34 03 00 00    	jl     401130 <_cudaGetErrorEnum(cudaError)+0x490>
  400dfc:	83 f8 1b             	cmp    eax,0x1b
  400dff:	0f 84 3f 03 00 00    	je     401144 <_cudaGetErrorEnum(cudaError)+0x4a4>
  400e05:	83 f8 1c             	cmp    eax,0x1c
  400e08:	0f 84 40 03 00 00    	je     40114e <_cudaGetErrorEnum(cudaError)+0x4ae>
  400e0e:	e9 0d 05 00 00       	jmp    401320 <_cudaGetErrorEnum(cudaError)+0x680>
  400e13:	83 f8 22             	cmp    eax,0x22
  400e16:	0f 84 6e 03 00 00    	je     40118a <_cudaGetErrorEnum(cudaError)+0x4ea>
  400e1c:	83 f8 22             	cmp    eax,0x22
  400e1f:	7f 29                	jg     400e4a <_cudaGetErrorEnum(cudaError)+0x1aa>
  400e21:	83 f8 1f             	cmp    eax,0x1f
  400e24:	0f 84 42 03 00 00    	je     40116c <_cudaGetErrorEnum(cudaError)+0x4cc>
  400e2a:	83 f8 1f             	cmp    eax,0x1f
  400e2d:	0f 8c 2f 03 00 00    	jl     401162 <_cudaGetErrorEnum(cudaError)+0x4c2>
  400e33:	83 f8 20             	cmp    eax,0x20
  400e36:	0f 84 3a 03 00 00    	je     401176 <_cudaGetErrorEnum(cudaError)+0x4d6>
  400e3c:	83 f8 21             	cmp    eax,0x21
  400e3f:	0f 84 3b 03 00 00    	je     401180 <_cudaGetErrorEnum(cudaError)+0x4e0>
  400e45:	e9 d6 04 00 00       	jmp    401320 <_cudaGetErrorEnum(cudaError)+0x680>
  400e4a:	83 f8 24             	cmp    eax,0x24
  400e4d:	0f 84 4b 03 00 00    	je     40119e <_cudaGetErrorEnum(cudaError)+0x4fe>
  400e53:	83 f8 24             	cmp    eax,0x24
  400e56:	0f 8c 38 03 00 00    	jl     401194 <_cudaGetErrorEnum(cudaError)+0x4f4>
  400e5c:	83 f8 25             	cmp    eax,0x25
  400e5f:	0f 84 43 03 00 00    	je     4011a8 <_cudaGetErrorEnum(cudaError)+0x508>
  400e65:	83 f8 26             	cmp    eax,0x26
  400e68:	0f 84 44 03 00 00    	je     4011b2 <_cudaGetErrorEnum(cudaError)+0x512>
  400e6e:	e9 ad 04 00 00       	jmp    401320 <_cudaGetErrorEnum(cudaError)+0x680>
  400e73:	83 f8 3d             	cmp    eax,0x3d
  400e76:	0f 84 08 04 00 00    	je     401284 <_cudaGetErrorEnum(cudaError)+0x5e4>
  400e7c:	83 f8 3d             	cmp    eax,0x3d
  400e7f:	0f 8f ce 00 00 00    	jg     400f53 <_cudaGetErrorEnum(cudaError)+0x2b3>
  400e85:	83 f8 31             	cmp    eax,0x31
  400e88:	0f 84 92 03 00 00    	je     401220 <_cudaGetErrorEnum(cudaError)+0x580>
  400e8e:	83 f8 31             	cmp    eax,0x31
  400e91:	7f 60                	jg     400ef3 <_cudaGetErrorEnum(cudaError)+0x253>
  400e93:	83 f8 2c             	cmp    eax,0x2c
  400e96:	0f 84 52 03 00 00    	je     4011ee <_cudaGetErrorEnum(cudaError)+0x54e>
  400e9c:	83 f8 2c             	cmp    eax,0x2c
  400e9f:	7f 29                	jg     400eca <_cudaGetErrorEnum(cudaError)+0x22a>
  400ea1:	83 f8 29             	cmp    eax,0x29
  400ea4:	0f 84 26 03 00 00    	je     4011d0 <_cudaGetErrorEnum(cudaError)+0x530>
  400eaa:	83 f8 29             	cmp    eax,0x29
  400ead:	0f 8c 13 03 00 00    	jl     4011c6 <_cudaGetErrorEnum(cudaError)+0x526>
  400eb3:	83 f8 2a             	cmp    eax,0x2a
  400eb6:	0f 84 1e 03 00 00    	je     4011da <_cudaGetErrorEnum(cudaError)+0x53a>
  400ebc:	83 f8 2b             	cmp    eax,0x2b
  400ebf:	0f 84 1f 03 00 00    	je     4011e4 <_cudaGetErrorEnum(cudaError)+0x544>
  400ec5:	e9 56 04 00 00       	jmp    401320 <_cudaGetErrorEnum(cudaError)+0x680>
  400eca:	83 f8 2e             	cmp    eax,0x2e
  400ecd:	0f 84 2f 03 00 00    	je     401202 <_cudaGetErrorEnum(cudaError)+0x562>
  400ed3:	83 f8 2e             	cmp    eax,0x2e
  400ed6:	0f 8c 1c 03 00 00    	jl     4011f8 <_cudaGetErrorEnum(cudaError)+0x558>
  400edc:	83 f8 2f             	cmp    eax,0x2f
  400edf:	0f 84 27 03 00 00    	je     40120c <_cudaGetErrorEnum(cudaError)+0x56c>
  400ee5:	83 f8 30             	cmp    eax,0x30
  400ee8:	0f 84 28 03 00 00    	je     401216 <_cudaGetErrorEnum(cudaError)+0x576>
  400eee:	e9 2d 04 00 00       	jmp    401320 <_cudaGetErrorEnum(cudaError)+0x680>
  400ef3:	83 f8 38             	cmp    eax,0x38
  400ef6:	0f 84 56 03 00 00    	je     401252 <_cudaGetErrorEnum(cudaError)+0x5b2>
  400efc:	83 f8 38             	cmp    eax,0x38
  400eff:	7f 29                	jg     400f2a <_cudaGetErrorEnum(cudaError)+0x28a>
  400f01:	83 f8 33             	cmp    eax,0x33
  400f04:	0f 84 2a 03 00 00    	je     401234 <_cudaGetErrorEnum(cudaError)+0x594>
  400f0a:	83 f8 33             	cmp    eax,0x33
  400f0d:	0f 8c 17 03 00 00    	jl     40122a <_cudaGetErrorEnum(cudaError)+0x58a>
  400f13:	83 f8 36             	cmp    eax,0x36
  400f16:	0f 84 22 03 00 00    	je     40123e <_cudaGetErrorEnum(cudaError)+0x59e>
  400f1c:	83 f8 37             	cmp    eax,0x37
  400f1f:	0f 84 23 03 00 00    	je     401248 <_cudaGetErrorEnum(cudaError)+0x5a8>
  400f25:	e9 f6 03 00 00       	jmp    401320 <_cudaGetErrorEnum(cudaError)+0x680>
  400f2a:	83 f8 3a             	cmp    eax,0x3a
  400f2d:	0f 84 33 03 00 00    	je     401266 <_cudaGetErrorEnum(cudaError)+0x5c6>
  400f33:	83 f8 3a             	cmp    eax,0x3a
  400f36:	0f 8c 20 03 00 00    	jl     40125c <_cudaGetErrorEnum(cudaError)+0x5bc>
  400f3c:	83 f8 3b             	cmp    eax,0x3b
  400f3f:	0f 84 2b 03 00 00    	je     401270 <_cudaGetErrorEnum(cudaError)+0x5d0>
  400f45:	83 f8 3c             	cmp    eax,0x3c
  400f48:	0f 84 2c 03 00 00    	je     40127a <_cudaGetErrorEnum(cudaError)+0x5da>
  400f4e:	e9 cd 03 00 00       	jmp    401320 <_cudaGetErrorEnum(cudaError)+0x680>
  400f53:	83 f8 47             	cmp    eax,0x47
  400f56:	0f 84 77 03 00 00    	je     4012d3 <_cudaGetErrorEnum(cudaError)+0x633>
  400f5c:	83 f8 47             	cmp    eax,0x47
  400f5f:	7f 60                	jg     400fc1 <_cudaGetErrorEnum(cudaError)+0x321>
  400f61:	83 f8 42             	cmp    eax,0x42
  400f64:	0f 84 46 03 00 00    	je     4012b0 <_cudaGetErrorEnum(cudaError)+0x610>
  400f6a:	83 f8 42             	cmp    eax,0x42
  400f6d:	7f 29                	jg     400f98 <_cudaGetErrorEnum(cudaError)+0x2f8>
  400f6f:	83 f8 3f             	cmp    eax,0x3f
  400f72:	0f 84 20 03 00 00    	je     401298 <_cudaGetErrorEnum(cudaError)+0x5f8>
  400f78:	83 f8 3f             	cmp    eax,0x3f
  400f7b:	0f 8c 0d 03 00 00    	jl     40128e <_cudaGetErrorEnum(cudaError)+0x5ee>
  400f81:	83 f8 40             	cmp    eax,0x40
  400f84:	0f 84 18 03 00 00    	je     4012a2 <_cudaGetErrorEnum(cudaError)+0x602>
  400f8a:	83 f8 41             	cmp    eax,0x41
  400f8d:	0f 84 16 03 00 00    	je     4012a9 <_cudaGetErrorEnum(cudaError)+0x609>
  400f93:	e9 88 03 00 00       	jmp    401320 <_cudaGetErrorEnum(cudaError)+0x680>
  400f98:	83 f8 44             	cmp    eax,0x44
  400f9b:	0f 84 1d 03 00 00    	je     4012be <_cudaGetErrorEnum(cudaError)+0x61e>
  400fa1:	83 f8 44             	cmp    eax,0x44
  400fa4:	0f 8c 0d 03 00 00    	jl     4012b7 <_cudaGetErrorEnum(cudaError)+0x617>
  400faa:	83 f8 45             	cmp    eax,0x45
  400fad:	0f 84 12 03 00 00    	je     4012c5 <_cudaGetErrorEnum(cudaError)+0x625>
  400fb3:	83 f8 46             	cmp    eax,0x46
  400fb6:	0f 84 10 03 00 00    	je     4012cc <_cudaGetErrorEnum(cudaError)+0x62c>
  400fbc:	e9 5f 03 00 00       	jmp    401320 <_cudaGetErrorEnum(cudaError)+0x680>
  400fc1:	83 f8 4c             	cmp    eax,0x4c
  400fc4:	0f 84 2c 03 00 00    	je     4012f6 <_cudaGetErrorEnum(cudaError)+0x656>
  400fca:	83 f8 4c             	cmp    eax,0x4c
  400fcd:	7f 29                	jg     400ff8 <_cudaGetErrorEnum(cudaError)+0x358>
  400fcf:	83 f8 49             	cmp    eax,0x49
  400fd2:	0f 84 09 03 00 00    	je     4012e1 <_cudaGetErrorEnum(cudaError)+0x641>
  400fd8:	83 f8 49             	cmp    eax,0x49
  400fdb:	0f 8c f9 02 00 00    	jl     4012da <_cudaGetErrorEnum(cudaError)+0x63a>
  400fe1:	83 f8 4a             	cmp    eax,0x4a
  400fe4:	0f 84 fe 02 00 00    	je     4012e8 <_cudaGetErrorEnum(cudaError)+0x648>
  400fea:	83 f8 4b             	cmp    eax,0x4b
  400fed:	0f 84 fc 02 00 00    	je     4012ef <_cudaGetErrorEnum(cudaError)+0x64f>
  400ff3:	e9 28 03 00 00       	jmp    401320 <_cudaGetErrorEnum(cudaError)+0x680>
  400ff8:	83 f8 4f             	cmp    eax,0x4f
  400ffb:	0f 84 0a 03 00 00    	je     40130b <_cudaGetErrorEnum(cudaError)+0x66b>
  401001:	83 f8 4f             	cmp    eax,0x4f
  401004:	7f 17                	jg     40101d <_cudaGetErrorEnum(cudaError)+0x37d>
  401006:	83 f8 4d             	cmp    eax,0x4d
  401009:	0f 84 ee 02 00 00    	je     4012fd <_cudaGetErrorEnum(cudaError)+0x65d>
  40100f:	83 f8 4e             	cmp    eax,0x4e
  401012:	0f 84 ec 02 00 00    	je     401304 <_cudaGetErrorEnum(cudaError)+0x664>
  401018:	e9 03 03 00 00       	jmp    401320 <_cudaGetErrorEnum(cudaError)+0x680>
  40101d:	83 f8 7f             	cmp    eax,0x7f
  401020:	0f 84 ec 02 00 00    	je     401312 <_cudaGetErrorEnum(cudaError)+0x672>
  401026:	3d 10 27 00 00       	cmp    eax,0x2710
  40102b:	0f 84 e8 02 00 00    	je     401319 <_cudaGetErrorEnum(cudaError)+0x679>
  401031:	e9 ea 02 00 00       	jmp    401320 <_cudaGetErrorEnum(cudaError)+0x680>
  401036:	b8 20 18 40 00       	mov    eax,0x401820
  40103b:	e9 e5 02 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401040:	b8 2c 18 40 00       	mov    eax,0x40182c
  401045:	e9 db 02 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  40104a:	b8 4a 18 40 00       	mov    eax,0x40184a
  40104f:	e9 d1 02 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401054:	b8 64 18 40 00       	mov    eax,0x401864
  401059:	e9 c7 02 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  40105e:	b8 81 18 40 00       	mov    eax,0x401881
  401063:	e9 bd 02 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401068:	b8 98 18 40 00       	mov    eax,0x401898
  40106d:	e9 b3 02 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401072:	b8 b4 18 40 00       	mov    eax,0x4018b4
  401077:	e9 a9 02 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  40107c:	b8 cb 18 40 00       	mov    eax,0x4018cb
  401081:	e9 9f 02 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401086:	b8 f0 18 40 00       	mov    eax,0x4018f0
  40108b:	e9 95 02 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401090:	b8 0f 19 40 00       	mov    eax,0x40190f
  401095:	e9 8b 02 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  40109a:	b8 2d 19 40 00       	mov    eax,0x40192d
  40109f:	e9 81 02 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4010a4:	b8 44 19 40 00       	mov    eax,0x401944
  4010a9:	e9 77 02 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4010ae:	b8 5a 19 40 00       	mov    eax,0x40195a
  4010b3:	e9 6d 02 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4010b8:	b8 75 19 40 00       	mov    eax,0x401975
  4010bd:	e9 63 02 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4010c2:	b8 90 19 40 00       	mov    eax,0x401990
  4010c7:	e9 59 02 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4010cc:	b8 b0 19 40 00       	mov    eax,0x4019b0
  4010d1:	e9 4f 02 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4010d6:	b8 d1 19 40 00       	mov    eax,0x4019d1
  4010db:	e9 45 02 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4010e0:	b8 ed 19 40 00       	mov    eax,0x4019ed
  4010e5:	e9 3b 02 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4010ea:	b8 0b 1a 40 00       	mov    eax,0x401a0b
  4010ef:	e9 31 02 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4010f4:	b8 28 1a 40 00       	mov    eax,0x401a28
  4010f9:	e9 27 02 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4010fe:	b8 48 1a 40 00       	mov    eax,0x401a48
  401103:	e9 1d 02 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401108:	b8 70 1a 40 00       	mov    eax,0x401a70
  40110d:	e9 13 02 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401112:	b8 90 1a 40 00       	mov    eax,0x401a90
  401117:	e9 09 02 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  40111c:	b8 ab 1a 40 00       	mov    eax,0x401aab
  401121:	e9 ff 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401126:	b8 c7 1a 40 00       	mov    eax,0x401ac7
  40112b:	e9 f5 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401130:	b8 e0 1a 40 00       	mov    eax,0x401ae0
  401135:	e9 eb 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  40113a:	b8 fe 1a 40 00       	mov    eax,0x401afe
  40113f:	e9 e1 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401144:	b8 1c 1b 40 00       	mov    eax,0x401b1c
  401149:	e9 d7 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  40114e:	b8 38 1b 40 00       	mov    eax,0x401b38
  401153:	e9 cd 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401158:	b8 56 1b 40 00       	mov    eax,0x401b56
  40115d:	e9 c3 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401162:	b8 6f 1b 40 00       	mov    eax,0x401b6f
  401167:	e9 b9 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  40116c:	b8 80 1b 40 00       	mov    eax,0x401b80
  401171:	e9 af 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401176:	b8 9b 1b 40 00       	mov    eax,0x401b9b
  40117b:	e9 a5 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401180:	b8 b8 1b 40 00       	mov    eax,0x401bb8
  401185:	e9 9b 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  40118a:	b8 d7 1b 40 00       	mov    eax,0x401bd7
  40118f:	e9 91 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401194:	b8 e9 1b 40 00       	mov    eax,0x401be9
  401199:	e9 87 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  40119e:	b8 05 1c 40 00       	mov    eax,0x401c05
  4011a3:	e9 7d 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4011a8:	b8 21 1c 40 00       	mov    eax,0x401c21
  4011ad:	e9 73 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4011b2:	b8 39 1c 40 00       	mov    eax,0x401c39
  4011b7:	e9 69 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4011bc:	b8 4b 1c 40 00       	mov    eax,0x401c4b
  4011c1:	e9 5f 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4011c6:	b8 68 1c 40 00       	mov    eax,0x401c68
  4011cb:	e9 55 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4011d0:	b8 90 1c 40 00       	mov    eax,0x401c90
  4011d5:	e9 4b 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4011da:	b8 b0 1c 40 00       	mov    eax,0x401cb0
  4011df:	e9 41 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4011e4:	b8 d0 1c 40 00       	mov    eax,0x401cd0
  4011e9:	e9 37 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4011ee:	b8 ef 1c 40 00       	mov    eax,0x401cef
  4011f3:	e9 2d 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4011f8:	b8 0d 1d 40 00       	mov    eax,0x401d0d
  4011fd:	e9 23 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401202:	b8 2b 1d 40 00       	mov    eax,0x401d2b
  401207:	e9 19 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  40120c:	b8 47 1d 40 00       	mov    eax,0x401d47
  401211:	e9 0f 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401216:	b8 68 1d 40 00       	mov    eax,0x401d68
  40121b:	e9 05 01 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401220:	b8 88 1d 40 00       	mov    eax,0x401d88
  401225:	e9 fb 00 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  40122a:	b8 b0 1d 40 00       	mov    eax,0x401db0
  40122f:	e9 f1 00 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401234:	b8 d2 1d 40 00       	mov    eax,0x401dd2
  401239:	e9 e7 00 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  40123e:	b8 f0 1d 40 00       	mov    eax,0x401df0
  401243:	e9 dd 00 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401248:	b8 0c 1e 40 00       	mov    eax,0x401e0c
  40124d:	e9 d3 00 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401252:	b8 28 1e 40 00       	mov    eax,0x401e28
  401257:	e9 c9 00 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  40125c:	b8 48 1e 40 00       	mov    eax,0x401e48
  401261:	e9 bf 00 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401266:	b8 68 1e 40 00       	mov    eax,0x401e68
  40126b:	e9 b5 00 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401270:	b8 88 1e 40 00       	mov    eax,0x401e88
  401275:	e9 ab 00 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  40127a:	b8 98 1e 40 00       	mov    eax,0x401e98
  40127f:	e9 a1 00 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401284:	b8 b0 1e 40 00       	mov    eax,0x401eb0
  401289:	e9 97 00 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  40128e:	b8 d8 1e 40 00       	mov    eax,0x401ed8
  401293:	e9 8d 00 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401298:	b8 f9 1e 40 00       	mov    eax,0x401ef9
  40129d:	e9 83 00 00 00       	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4012a2:	b8 18 1f 40 00       	mov    eax,0x401f18
  4012a7:	eb 7c                	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4012a9:	b8 38 1f 40 00       	mov    eax,0x401f38
  4012ae:	eb 75                	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4012b0:	b8 58 1f 40 00       	mov    eax,0x401f58
  4012b5:	eb 6e                	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4012b7:	b8 75 1f 40 00       	mov    eax,0x401f75
  4012bc:	eb 67                	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4012be:	b8 93 1f 40 00       	mov    eax,0x401f93
  4012c3:	eb 60                	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4012c5:	b8 b0 1f 40 00       	mov    eax,0x401fb0
  4012ca:	eb 59                	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4012cc:	b8 d4 1f 40 00       	mov    eax,0x401fd4
  4012d1:	eb 52                	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4012d3:	b8 ea 1f 40 00       	mov    eax,0x401fea
  4012d8:	eb 4b                	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4012da:	b8 00 20 40 00       	mov    eax,0x402000
  4012df:	eb 44                	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4012e1:	b8 1c 20 40 00       	mov    eax,0x40201c
  4012e6:	eb 3d                	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4012e8:	b8 38 20 40 00       	mov    eax,0x402038
  4012ed:	eb 36                	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4012ef:	b8 53 20 40 00       	mov    eax,0x402053
  4012f4:	eb 2f                	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4012f6:	b8 70 20 40 00       	mov    eax,0x402070
  4012fb:	eb 28                	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  4012fd:	b8 83 20 40 00       	mov    eax,0x402083
  401302:	eb 21                	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401304:	b8 9b 20 40 00       	mov    eax,0x40209b
  401309:	eb 1a                	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  40130b:	b8 b0 20 40 00       	mov    eax,0x4020b0
  401310:	eb 13                	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401312:	b8 d0 20 40 00       	mov    eax,0x4020d0
  401317:	eb 0c                	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401319:	b8 e8 20 40 00       	mov    eax,0x4020e8
  40131e:	eb 05                	jmp    401325 <_cudaGetErrorEnum(cudaError)+0x685>
  401320:	b8 00 21 40 00       	mov    eax,0x402100
  401325:	5d                   	pop    rbp
  401326:	c3                   	ret    

0000000000401327 <main>:
  401327:	55                   	push   rbp
  401328:	48 89 e5             	mov    rbp,rsp
  40132b:	48 83 ec 60          	sub    rsp,0x60
  40132f:	89 7d ac             	mov    DWORD PTR [rbp-0x54],edi
  401332:	48 89 75 a0          	mov    QWORD PTR [rbp-0x60],rsi
  401336:	64 48 8b 04 25 28 00 	mov    rax,QWORD PTR fs:0x28
  40133d:	00 00 
  40133f:	48 89 45 f8          	mov    QWORD PTR [rbp-0x8],rax
  401343:	31 c0                	xor    eax,eax
  401345:	ba 10 00 00 00       	mov    edx,0x10
  40134a:	48 8d 45 c0          	lea    rax,[rbp-0x40]
  40134e:	48 89 d6             	mov    rsi,rdx
  401351:	48 89 c7             	mov    rdi,rax
  401354:	e8 ce 02 00 00       	call   401627 <cudaError cudaMalloc<unsigned int>(unsigned int**, unsigned long)>
  401359:	b9 1e 00 00 00       	mov    ecx,0x1e
  40135e:	ba 0a 21 40 00       	mov    edx,0x40210a
  401363:	be 18 21 40 00       	mov    esi,0x402118
  401368:	89 c7                	mov    edi,eax
  40136a:	e8 79 03 00 00       	call   4016e8 <void check<cudaError>(cudaError, char const*, char const*, int)>
  40136f:	ba 10 00 00 00       	mov    edx,0x10
  401374:	48 8b 45 c0          	mov    rax,QWORD PTR [rbp-0x40]
  401378:	be 00 00 00 00       	mov    esi,0x0
  40137d:	48 89 c7             	mov    rdi,rax
  401380:	e8 ab f6 ff ff       	call   400a30 <cudaMemset@plt>
  401385:	b9 1f 00 00 00       	mov    ecx,0x1f
  40138a:	ba 0a 21 40 00       	mov    edx,0x40210a
  40138f:	be 50 21 40 00       	mov    esi,0x402150
  401394:	89 c7                	mov    edi,eax
  401396:	e8 4d 03 00 00       	call   4016e8 <void check<cudaError>(cudaError, char const*, char const*, int)>
  40139b:	b8 10 00 00 00       	mov    eax,0x10
  4013a0:	48 89 c7             	mov    rdi,rax
  4013a3:	e8 f8 f6 ff ff       	call   400aa0 <malloc@plt>
  4013a8:	48 89 45 c8          	mov    QWORD PTR [rbp-0x38],rax
  4013ac:	48 8d 45 e0          	lea    rax,[rbp-0x20]
  4013b0:	b9 01 00 00 00       	mov    ecx,0x1
  4013b5:	ba 01 00 00 00       	mov    edx,0x1
  4013ba:	be 02 00 00 00       	mov    esi,0x2
  4013bf:	48 89 c7             	mov    rdi,rax
  4013c2:	e8 f1 02 00 00       	call   4016b8 <dim3::dim3(unsigned int, unsigned int, unsigned int)>
  4013c7:	48 8d 45 d0          	lea    rax,[rbp-0x30]
  4013cb:	b9 01 00 00 00       	mov    ecx,0x1
  4013d0:	ba 01 00 00 00       	mov    edx,0x1
  4013d5:	be 02 00 00 00       	mov    esi,0x2
  4013da:	48 89 c7             	mov    rdi,rax
  4013dd:	e8 d6 02 00 00       	call   4016b8 <dim3::dim3(unsigned int, unsigned int, unsigned int)>
  4013e2:	48 8b 55 e0          	mov    rdx,QWORD PTR [rbp-0x20]
  4013e6:	8b 4d e8             	mov    ecx,DWORD PTR [rbp-0x18]
  4013e9:	48 8b 75 d0          	mov    rsi,QWORD PTR [rbp-0x30]
  4013ed:	8b 45 d8             	mov    eax,DWORD PTR [rbp-0x28]
  4013f0:	41 b9 00 00 00 00    	mov    r9d,0x0
  4013f6:	41 b8 00 00 00 00    	mov    r8d,0x0
  4013fc:	48 89 f7             	mov    rdi,rsi
  4013ff:	89 c6                	mov    esi,eax
  401401:	e8 6a f6 ff ff       	call   400a70 <cudaConfigureCall@plt>
  401406:	85 c0                	test   eax,eax
  401408:	75 0c                	jne    401416 <main+0xef>
  40140a:	48 8b 45 c0          	mov    rax,QWORD PTR [rbp-0x40]
  40140e:	48 89 c7             	mov    rdi,rax
  401411:	e8 62 01 00 00       	call   401578 <tstfun(unsigned int volatile*)>
  401416:	e8 c5 f6 ff ff       	call   400ae0 <cudaDeviceSynchronize@plt>
  40141b:	b9 23 00 00 00       	mov    ecx,0x23
  401420:	ba 0a 21 40 00       	mov    edx,0x40210a
  401425:	be 8a 21 40 00       	mov    esi,0x40218a
  40142a:	89 c7                	mov    edi,eax
  40142c:	e8 b7 02 00 00       	call   4016e8 <void check<cudaError>(cudaError, char const*, char const*, int)>
  401431:	ba 10 00 00 00       	mov    edx,0x10
  401436:	48 8b 75 c0          	mov    rsi,QWORD PTR [rbp-0x40]
  40143a:	48 8b 45 c8          	mov    rax,QWORD PTR [rbp-0x38]
  40143e:	b9 02 00 00 00       	mov    ecx,0x2
  401443:	48 89 c7             	mov    rdi,rax
  401446:	e8 f5 f6 ff ff       	call   400b40 <cudaMemcpy@plt>
  40144b:	b9 24 00 00 00       	mov    ecx,0x24
  401450:	ba 0a 21 40 00       	mov    edx,0x40210a
  401455:	be a8 21 40 00       	mov    esi,0x4021a8
  40145a:	89 c7                	mov    edi,eax
  40145c:	e8 87 02 00 00       	call   4016e8 <void check<cudaError>(cudaError, char const*, char const*, int)>
  401461:	c7 45 bc 00 00 00 00 	mov    DWORD PTR [rbp-0x44],0x0
  401468:	eb 51                	jmp    4014bb <main+0x194>
  40146a:	8b 45 bc             	mov    eax,DWORD PTR [rbp-0x44]
  40146d:	48 98                	cdqe   
  40146f:	48 8d 14 85 00 00 00 	lea    rdx,[rax*4+0x0]
  401476:	00 
  401477:	48 8b 45 c8          	mov    rax,QWORD PTR [rbp-0x38]
  40147b:	48 01 d0             	add    rax,rdx
  40147e:	8b 00                	mov    eax,DWORD PTR [rax]
  401480:	83 f8 0c             	cmp    eax,0xc
  401483:	74 32                	je     4014b7 <main+0x190>
  401485:	8b 45 bc             	mov    eax,DWORD PTR [rbp-0x44]
  401488:	48 98                	cdqe   
  40148a:	48 8d 14 85 00 00 00 	lea    rdx,[rax*4+0x0]
  401491:	00 
  401492:	48 8b 45 c8          	mov    rax,QWORD PTR [rbp-0x38]
  401496:	48 01 d0             	add    rax,rdx
  401499:	8b 08                	mov    ecx,DWORD PTR [rax]
  40149b:	48 8b 05 3e 2c 20 00 	mov    rax,QWORD PTR [rip+0x202c3e]        # 6040e0 <stderr@@GLIBC_2.2.5>
  4014a2:	8b 55 bc             	mov    edx,DWORD PTR [rbp-0x44]
  4014a5:	be 02 22 40 00       	mov    esi,0x402202
  4014aa:	48 89 c7             	mov    rdi,rax
  4014ad:	b8 00 00 00 00       	mov    eax,0x0
  4014b2:	e8 f9 f5 ff ff       	call   400ab0 <fprintf@plt>
  4014b7:	83 45 bc 01          	add    DWORD PTR [rbp-0x44],0x1
  4014bb:	83 7d bc 03          	cmp    DWORD PTR [rbp-0x44],0x3
  4014bf:	7e a9                	jle    40146a <main+0x143>
  4014c1:	bf 14 22 40 00       	mov    edi,0x402214
  4014c6:	e8 05 f6 ff ff       	call   400ad0 <puts@plt>
  4014cb:	b8 00 00 00 00       	mov    eax,0x0
  4014d0:	48 8b 4d f8          	mov    rcx,QWORD PTR [rbp-0x8]
  4014d4:	64 48 33 0c 25 28 00 	xor    rcx,QWORD PTR fs:0x28
  4014db:	00 00 
  4014dd:	74 05                	je     4014e4 <main+0x1bd>
  4014df:	e8 9c f5 ff ff       	call   400a80 <__stack_chk_fail@plt>
  4014e4:	c9                   	leave  
  4014e5:	c3                   	ret    

00000000004014e6 <____nv_dummy_param_ref(void*)>:
  4014e6:	55                   	push   rbp
  4014e7:	48 89 e5             	mov    rbp,rsp
  4014ea:	48 89 7d f8          	mov    QWORD PTR [rbp-0x8],rdi
  4014ee:	48 8b 45 f8          	mov    rax,QWORD PTR [rbp-0x8]
  4014f2:	48 89 05 27 2c 20 00 	mov    QWORD PTR [rip+0x202c27],rax        # 604120 <____nv_dummy_param_ref(void*)::__ref>
  4014f9:	5d                   	pop    rbp
  4014fa:	c3                   	ret    

00000000004014fb <__cudaUnregisterBinaryUtil()>:
  4014fb:	55                   	push   rbp
  4014fc:	48 89 e5             	mov    rbp,rsp
  4014ff:	bf 18 41 60 00       	mov    edi,0x604118
  401504:	e8 dd ff ff ff       	call   4014e6 <____nv_dummy_param_ref(void*)>
  401509:	48 8b 05 08 2c 20 00 	mov    rax,QWORD PTR [rip+0x202c08]        # 604118 <__cudaFatCubinHandle>
  401510:	48 89 c7             	mov    rdi,rax
  401513:	e8 e8 f5 ff ff       	call   400b00 <__cudaUnregisterFatBinary@plt>
  401518:	5d                   	pop    rbp
  401519:	c3                   	ret    

000000000040151a <__nv_init_managed_rt_with_module(void**)>:
  40151a:	55                   	push   rbp
  40151b:	48 89 e5             	mov    rbp,rsp
  40151e:	48 83 ec 10          	sub    rsp,0x10
  401522:	48 89 7d f8          	mov    QWORD PTR [rbp-0x8],rdi
  401526:	48 8b 45 f8          	mov    rax,QWORD PTR [rbp-0x8]
  40152a:	48 89 c7             	mov    rdi,rax
  40152d:	e8 1e f5 ff ff       	call   400a50 <__cudaInitModule@plt>
  401532:	c9                   	leave  
  401533:	c3                   	ret    

0000000000401534 <__device_stub__Z6tstfunPVj(unsigned int volatile*)>:
  401534:	55                   	push   rbp
  401535:	48 89 e5             	mov    rbp,rsp
  401538:	48 83 ec 10          	sub    rsp,0x10
  40153c:	48 89 7d f8          	mov    QWORD PTR [rbp-0x8],rdi
  401540:	48 8d 45 f8          	lea    rax,[rbp-0x8]
  401544:	ba 00 00 00 00       	mov    edx,0x0
  401549:	be 08 00 00 00       	mov    esi,0x8
  40154e:	48 89 c7             	mov    rdi,rax
  401551:	e8 ca f4 ff ff       	call   400a20 <cudaSetupArgument@plt>
  401556:	85 c0                	test   eax,eax
  401558:	0f 95 c0             	setne  al
  40155b:	84 c0                	test   al,al
  40155d:	74 02                	je     401561 <__device_stub__Z6tstfunPVj(unsigned int volatile*)+0x2d>
  40155f:	eb 15                	jmp    401576 <__device_stub__Z6tstfunPVj(unsigned int volatile*)+0x42>
  401561:	48 c7 05 bc 2b 20 00 	mov    QWORD PTR [rip+0x202bbc],0x401578        # 604128 <__device_stub__Z6tstfunPVj(unsigned int volatile*)::__f>
  401568:	78 15 40 00 
  40156c:	bf 78 15 40 00       	mov    edi,0x401578
  401571:	e8 d6 00 00 00       	call   40164c <cudaError cudaLaunch<char>(char*)>
  401576:	c9                   	leave  
  401577:	c3                   	ret    

0000000000401578 <tstfun(unsigned int volatile*)>:
  401578:	55                   	push   rbp
  401579:	48 89 e5             	mov    rbp,rsp
  40157c:	48 83 ec 10          	sub    rsp,0x10
  401580:	48 89 7d f8          	mov    QWORD PTR [rbp-0x8],rdi
  401584:	48 8b 45 f8          	mov    rax,QWORD PTR [rbp-0x8]
  401588:	48 89 c7             	mov    rdi,rax
  40158b:	e8 a4 ff ff ff       	call   401534 <__device_stub__Z6tstfunPVj(unsigned int volatile*)>
  401590:	c9                   	leave  
  401591:	c3                   	ret    

0000000000401592 <__nv_cudaEntityRegisterCallback(void**)>:
  401592:	55                   	push   rbp
  401593:	48 89 e5             	mov    rbp,rsp
  401596:	48 83 ec 10          	sub    rsp,0x10
  40159a:	48 89 7d f8          	mov    QWORD PTR [rbp-0x8],rdi
  40159e:	48 8b 45 f8          	mov    rax,QWORD PTR [rbp-0x8]
  4015a2:	48 89 05 87 2b 20 00 	mov    QWORD PTR [rip+0x202b87],rax        # 604130 <__nv_cudaEntityRegisterCallback(void**)::__ref>
  4015a9:	48 8b 45 f8          	mov    rax,QWORD PTR [rbp-0x8]
  4015ad:	48 89 c7             	mov    rdi,rax
  4015b0:	e8 d6 f6 ff ff       	call   400c8b <__nv_save_fatbinhandle_for_managed_rt(void**)>
  4015b5:	48 8b 45 f8          	mov    rax,QWORD PTR [rbp-0x8]
  4015b9:	6a 00                	push   0x0
  4015bb:	6a 00                	push   0x0
  4015bd:	6a 00                	push   0x0
  4015bf:	6a 00                	push   0x0
  4015c1:	41 b9 00 00 00 00    	mov    r9d,0x0
  4015c7:	41 b8 ff ff ff ff    	mov    r8d,0xffffffff
  4015cd:	b9 1d 22 40 00       	mov    ecx,0x40221d
  4015d2:	ba 1d 22 40 00       	mov    edx,0x40221d
  4015d7:	be 78 15 40 00       	mov    esi,0x401578
  4015dc:	48 89 c7             	mov    rdi,rax
  4015df:	e8 5c f4 ff ff       	call   400a40 <__cudaRegisterFunction@plt>
  4015e4:	48 83 c4 20          	add    rsp,0x20
  4015e8:	c9                   	leave  
  4015e9:	c3                   	ret    

00000000004015ea <__sti____cudaRegisterAll_39_tmpxft_00000a69_00000000_9_tst1_cpp1_ii_442ddbb3()>:
  4015ea:	55                   	push   rbp
  4015eb:	48 89 e5             	mov    rbp,rsp
  4015ee:	48 83 ec 10          	sub    rsp,0x10
  4015f2:	bf e8 30 40 00       	mov    edi,0x4030e8
  4015f7:	e8 14 f5 ff ff       	call   400b10 <__cudaRegisterFatBinary@plt>
  4015fc:	48 89 05 15 2b 20 00 	mov    QWORD PTR [rip+0x202b15],rax        # 604118 <__cudaFatCubinHandle>
  401603:	48 c7 45 f8 92 15 40 	mov    QWORD PTR [rbp-0x8],0x401592
  40160a:	00 
  40160b:	48 8b 15 06 2b 20 00 	mov    rdx,QWORD PTR [rip+0x202b06]        # 604118 <__cudaFatCubinHandle>
  401612:	48 8b 45 f8          	mov    rax,QWORD PTR [rbp-0x8]
  401616:	48 89 d7             	mov    rdi,rdx
  401619:	ff d0                	call   rax
  40161b:	bf fb 14 40 00       	mov    edi,0x4014fb
  401620:	e8 bb 01 00 00       	call   4017e0 <atexit>
  401625:	c9                   	leave  
  401626:	c3                   	ret    

0000000000401627 <cudaError cudaMalloc<unsigned int>(unsigned int**, unsigned long)>:
  401627:	55                   	push   rbp
  401628:	48 89 e5             	mov    rbp,rsp
  40162b:	48 83 ec 10          	sub    rsp,0x10
  40162f:	48 89 7d f8          	mov    QWORD PTR [rbp-0x8],rdi
  401633:	48 89 75 f0          	mov    QWORD PTR [rbp-0x10],rsi
  401637:	48 8b 55 f0          	mov    rdx,QWORD PTR [rbp-0x10]
  40163b:	48 8b 45 f8          	mov    rax,QWORD PTR [rbp-0x8]
  40163f:	48 89 d6             	mov    rsi,rdx
  401642:	48 89 c7             	mov    rdi,rax
  401645:	e8 e6 f4 ff ff       	call   400b30 <cudaMalloc@plt>
  40164a:	c9                   	leave  
  40164b:	c3                   	ret    

000000000040164c <cudaError cudaLaunch<char>(char*)>:
  40164c:	55                   	push   rbp
  40164d:	48 89 e5             	mov    rbp,rsp
  401650:	48 83 ec 10          	sub    rsp,0x10
  401654:	48 89 7d f8          	mov    QWORD PTR [rbp-0x8],rdi
  401658:	48 8b 45 f8          	mov    rax,QWORD PTR [rbp-0x8]
  40165c:	48 89 c7             	mov    rdi,rax
  40165f:	e8 8c f4 ff ff       	call   400af0 <cudaLaunch@plt>
  401664:	c9                   	leave  
  401665:	c3                   	ret    

0000000000401666 <__static_initialization_and_destruction_0(int, int)>:
  401666:	55                   	push   rbp
  401667:	48 89 e5             	mov    rbp,rsp
  40166a:	48 83 ec 10          	sub    rsp,0x10
  40166e:	89 7d fc             	mov    DWORD PTR [rbp-0x4],edi
  401671:	89 75 f8             	mov    DWORD PTR [rbp-0x8],esi
  401674:	83 7d fc 01          	cmp    DWORD PTR [rbp-0x4],0x1
  401678:	75 27                	jne    4016a1 <__static_initialization_and_destruction_0(int, int)+0x3b>
  40167a:	81 7d f8 ff ff 00 00 	cmp    DWORD PTR [rbp-0x8],0xffff
  401681:	75 1e                	jne    4016a1 <__static_initialization_and_destruction_0(int, int)+0x3b>
  401683:	bf 10 41 60 00       	mov    edi,0x604110
  401688:	e8 33 f4 ff ff       	call   400ac0 <std::ios_base::Init::Init()@plt>
  40168d:	ba c8 40 60 00       	mov    edx,0x6040c8
  401692:	be 10 41 60 00       	mov    esi,0x604110
  401697:	bf 60 0b 40 00       	mov    edi,0x400b60
  40169c:	e8 bf f3 ff ff       	call   400a60 <__cxa_atexit@plt>
  4016a1:	c9                   	leave  
  4016a2:	c3                   	ret    

00000000004016a3 <_GLOBAL__sub_I_main>:
  4016a3:	55                   	push   rbp
  4016a4:	48 89 e5             	mov    rbp,rsp
  4016a7:	be ff ff 00 00       	mov    esi,0xffff
  4016ac:	bf 01 00 00 00       	mov    edi,0x1
  4016b1:	e8 b0 ff ff ff       	call   401666 <__static_initialization_and_destruction_0(int, int)>
  4016b6:	5d                   	pop    rbp
  4016b7:	c3                   	ret    

00000000004016b8 <dim3::dim3(unsigned int, unsigned int, unsigned int)>:
  4016b8:	55                   	push   rbp
  4016b9:	48 89 e5             	mov    rbp,rsp
  4016bc:	48 89 7d f8          	mov    QWORD PTR [rbp-0x8],rdi
  4016c0:	89 75 f4             	mov    DWORD PTR [rbp-0xc],esi
  4016c3:	89 55 f0             	mov    DWORD PTR [rbp-0x10],edx
  4016c6:	89 4d ec             	mov    DWORD PTR [rbp-0x14],ecx
  4016c9:	48 8b 45 f8          	mov    rax,QWORD PTR [rbp-0x8]
  4016cd:	8b 55 f4             	mov    edx,DWORD PTR [rbp-0xc]
  4016d0:	89 10                	mov    DWORD PTR [rax],edx
  4016d2:	48 8b 45 f8          	mov    rax,QWORD PTR [rbp-0x8]
  4016d6:	8b 55 f0             	mov    edx,DWORD PTR [rbp-0x10]
  4016d9:	89 50 04             	mov    DWORD PTR [rax+0x4],edx
  4016dc:	48 8b 45 f8          	mov    rax,QWORD PTR [rbp-0x8]
  4016e0:	8b 55 ec             	mov    edx,DWORD PTR [rbp-0x14]
  4016e3:	89 50 08             	mov    DWORD PTR [rax+0x8],edx
  4016e6:	5d                   	pop    rbp
  4016e7:	c3                   	ret    

00000000004016e8 <void check<cudaError>(cudaError, char const*, char const*, int)>:
  4016e8:	55                   	push   rbp
  4016e9:	48 89 e5             	mov    rbp,rsp
  4016ec:	48 83 ec 20          	sub    rsp,0x20
  4016f0:	89 7d fc             	mov    DWORD PTR [rbp-0x4],edi
  4016f3:	48 89 75 f0          	mov    QWORD PTR [rbp-0x10],rsi
  4016f7:	48 89 55 e8          	mov    QWORD PTR [rbp-0x18],rdx
  4016fb:	89 4d f8             	mov    DWORD PTR [rbp-0x8],ecx
  4016fe:	83 7d fc 00          	cmp    DWORD PTR [rbp-0x4],0x0
  401702:	74 50                	je     401754 <void check<cudaError>(cudaError, char const*, char const*, int)+0x6c>
  401704:	8b 45 fc             	mov    eax,DWORD PTR [rbp-0x4]
  401707:	89 c7                	mov    edi,eax
  401709:	e8 92 f5 ff ff       	call   400ca0 <_cudaGetErrorEnum(cudaError)>
  40170e:	48 89 c7             	mov    rdi,rax
  401711:	48 8b 05 c8 29 20 00 	mov    rax,QWORD PTR [rip+0x2029c8]        # 6040e0 <stderr@@GLIBC_2.2.5>
  401718:	8b 75 fc             	mov    esi,DWORD PTR [rbp-0x4]
  40171b:	8b 4d f8             	mov    ecx,DWORD PTR [rbp-0x8]
  40171e:	48 8b 55 e8          	mov    rdx,QWORD PTR [rbp-0x18]
  401722:	48 83 ec 08          	sub    rsp,0x8
  401726:	ff 75 f0             	push   QWORD PTR [rbp-0x10]
  401729:	49 89 f9             	mov    r9,rdi
  40172c:	41 89 f0             	mov    r8d,esi
  40172f:	be 30 22 40 00       	mov    esi,0x402230
  401734:	48 89 c7             	mov    rdi,rax
  401737:	b8 00 00 00 00       	mov    eax,0x0
  40173c:	e8 6f f3 ff ff       	call   400ab0 <fprintf@plt>
  401741:	48 83 c4 10          	add    rsp,0x10
  401745:	e8 06 f4 ff ff       	call   400b50 <cudaDeviceReset@plt>
  40174a:	bf 01 00 00 00       	mov    edi,0x1
  40174f:	e8 3c f3 ff ff       	call   400a90 <exit@plt>
  401754:	c9                   	leave  
  401755:	c3                   	ret    
  401756:	66 2e 0f 1f 84 00 00 	nop    WORD PTR cs:[rax+rax*1+0x0]
  40175d:	00 00 00 

0000000000401760 <__libc_csu_init>:
  401760:	41 57                	push   r15
  401762:	41 56                	push   r14
  401764:	41 89 ff             	mov    r15d,edi
  401767:	41 55                	push   r13
  401769:	41 54                	push   r12
  40176b:	4c 8d 25 6e 26 20 00 	lea    r12,[rip+0x20266e]        # 603de0 <__frame_dummy_init_array_entry>
  401772:	55                   	push   rbp
  401773:	48 8d 2d 7e 26 20 00 	lea    rbp,[rip+0x20267e]        # 603df8 <__init_array_end>
  40177a:	53                   	push   rbx
  40177b:	49 89 f6             	mov    r14,rsi
  40177e:	49 89 d5             	mov    r13,rdx
  401781:	4c 29 e5             	sub    rbp,r12
  401784:	48 83 ec 08          	sub    rsp,0x8
  401788:	48 c1 fd 03          	sar    rbp,0x3
  40178c:	e8 57 f2 ff ff       	call   4009e8 <_init>
  401791:	48 85 ed             	test   rbp,rbp
  401794:	74 20                	je     4017b6 <__libc_csu_init+0x56>
  401796:	31 db                	xor    ebx,ebx
  401798:	0f 1f 84 00 00 00 00 	nop    DWORD PTR [rax+rax*1+0x0]
  40179f:	00 
  4017a0:	4c 89 ea             	mov    rdx,r13
  4017a3:	4c 89 f6             	mov    rsi,r14
  4017a6:	44 89 ff             	mov    edi,r15d
  4017a9:	41 ff 14 dc          	call   QWORD PTR [r12+rbx*8]
  4017ad:	48 83 c3 01          	add    rbx,0x1
  4017b1:	48 39 eb             	cmp    rbx,rbp
  4017b4:	75 ea                	jne    4017a0 <__libc_csu_init+0x40>
  4017b6:	48 83 c4 08          	add    rsp,0x8
  4017ba:	5b                   	pop    rbx
  4017bb:	5d                   	pop    rbp
  4017bc:	41 5c                	pop    r12
  4017be:	41 5d                	pop    r13
  4017c0:	41 5e                	pop    r14
  4017c2:	41 5f                	pop    r15
  4017c4:	c3                   	ret    
  4017c5:	90                   	nop
  4017c6:	66 2e 0f 1f 84 00 00 	nop    WORD PTR cs:[rax+rax*1+0x0]
  4017cd:	00 00 00 

00000000004017d0 <__libc_csu_fini>:
  4017d0:	f3 c3                	repz ret 
  4017d2:	66 2e 0f 1f 84 00 00 	nop    WORD PTR cs:[rax+rax*1+0x0]
  4017d9:	00 00 00 
  4017dc:	0f 1f 40 00          	nop    DWORD PTR [rax+0x0]

00000000004017e0 <atexit>:
  4017e0:	48 8d 05 e1 28 20 00 	lea    rax,[rip+0x2028e1]        # 6040c8 <__dso_handle>
  4017e7:	48 85 c0             	test   rax,rax
  4017ea:	74 14                	je     401800 <atexit+0x20>
  4017ec:	48 8b 10             	mov    rdx,QWORD PTR [rax]
  4017ef:	31 f6                	xor    esi,esi
  4017f1:	e9 6a f2 ff ff       	jmp    400a60 <__cxa_atexit@plt>
  4017f6:	66 2e 0f 1f 84 00 00 	nop    WORD PTR cs:[rax+rax*1+0x0]
  4017fd:	00 00 00 
  401800:	31 d2                	xor    edx,edx
  401802:	31 f6                	xor    esi,esi
  401804:	e9 57 f2 ff ff       	jmp    400a60 <__cxa_atexit@plt>

Disassembly of section .fini:

000000000040180c <_fini>:
  40180c:	48 83 ec 08          	sub    rsp,0x8
  401810:	48 83 c4 08          	add    rsp,0x8
  401814:	c3                   	ret    
