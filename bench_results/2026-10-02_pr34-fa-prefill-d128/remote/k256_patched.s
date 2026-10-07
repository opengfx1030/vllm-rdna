SYM:
.Lfunc_begin19:
	s_load_dwordx4 s[0:3], s[4:5], 0x20
.Ltmp10974:
	s_ashr_i32 s9, s8, 31
	s_lshl_b64 s[10:11], s[8:9], 2
	s_waitcnt lgkmcnt(0)
	s_add_u32 s0, s0, s10
.Ltmp10975:
	s_addc_u32 s1, s1, s11
	s_load_dwordx2 s[40:41], s[0:1], 0x0
.Ltmp10976:
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s9, s41, s40
.Ltmp10977:
	s_lshl_b32 s41, s6, 3
.Ltmp10978:
	s_cmp_ge_i32 s41, s9
.Ltmp10979:
	s_cbranch_scc1 .LBB19_115
.Ltmp10980:
	s_load_dwordx4 s[28:31], s[4:5], 0x78
.Ltmp10981:
	s_abs_i32 s13, s7
	s_waitcnt lgkmcnt(0)
	s_lshr_b32 s0, s28, 31
	s_add_i32 s0, s28, s0
	s_ashr_i32 s0, s0, 1
	s_abs_i32 s1, s0
	v_cvt_f32_u32_e32 v1, s1
	s_sub_i32 s12, 0, s1
	v_rcp_iflag_f32_e32 v1, v1
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_readfirstlane_b32 s6, v1
.Ltmp10982:
	s_mul_i32 s12, s12, s6
	s_mul_hi_u32 s12, s6, s12
	s_add_i32 s6, s6, s12
	s_xor_b32 s12, s7, s0
	s_mul_hi_u32 s6, s13, s6
	s_ashr_i32 s12, s12, 31
	s_mul_i32 s14, s6, s1
	s_sub_i32 s13, s13, s14
	s_add_i32 s14, s6, 1
	s_sub_i32 s15, s13, s1
	s_cmp_ge_u32 s13, s1
	s_cselect_b32 s6, s14, s6
	s_cselect_b32 s13, s15, s13
	s_add_i32 s14, s6, 1
	s_cmp_ge_u32 s13, s1
	s_cselect_b32 s1, s14, s6
	s_xor_b32 s1, s1, s12
	s_sub_i32 s6, s1, s12
.Ltmp10983:
	s_mul_i32 s0, s6, s0
	s_mul_i32 s1, s6, s28
	s_sub_i32 s0, s7, s0
	s_lshl1_add_u32 s28, s0, s1
.Ltmp10984:
	s_add_u32 s0, s2, s10
	s_addc_u32 s1, s3, s11
	s_sub_i32 s42, s9, s41
	s_load_dword s2, s[0:1], 0x0
.Ltmp10985:
	s_clause 0x2
	s_load_dwordx8 s[20:27], s[4:5], 0x0
	s_load_dwordx2 s[34:35], s[4:5], 0x68
	s_load_dword s11, s[4:5], 0x70
.Ltmp10986:
	s_min_i32 s33, s42, 8
.Ltmp10987:
	s_mov_b32 s0, -1
.Ltmp10988:
	s_waitcnt lgkmcnt(0)
	s_cmp_gt_i32 s2, 0
	s_cbranch_scc0 .LBB19_106
.Ltmp10989:
	v_cmp_gt_u32_e64 s0, 0x200, v0
.Ltmp10990:
	s_and_saveexec_b32 s3, s0
	s_cbranch_execz .LBB19_9
.Ltmp10991:
	v_lshrrev_b32_e32 v1, 5, v0
	v_lshlrev_b32_e32 v2, 4, v0
	v_mov_b32_e32 v9, v0
	s_mov_b32 s7, 0
	v_add3_u32 v4, s40, s41, v1
	v_and_b32_e32 v2, 0x1f0, v2
	v_lshlrev_b32_e32 v3, 9, v1
	v_cmp_le_i32_e32 vcc_lo, s42, v1
	v_mov_b32_e32 v1, 0
	v_mad_u64_u32 v[5:6], null, v4, s11, s[28:29]
	v_add_co_u32 v6, s1, s20, v2
	v_add3_u32 v7, 0x80, v3, v2
	v_add_co_ci_u32_e64 v8, null, s21, 0, s1
	s_inst_prefetch 0x1
	s_branch .LBB19_5
.Ltmp10992:
.Ltmp10993:
	.p2align	6
.LBB19_4:
	s_or_b32 exec_lo, exec_lo, s10
.Ltmp10994:
	v_add_nc_u32_e32 v2, 0x100, v9
.Ltmp10995:
	v_cmp_lt_u32_e64 s1, 0xff, v9
	v_mov_b32_e32 v9, v2
.Ltmp10996:
	s_or_b32 s7, s1, s7
	s_andn2_b32 exec_lo, exec_lo, s7
	s_cbranch_execz .LBB19_9
.Ltmp10997:
.LBB19_5:
	v_lshrrev_b32_e32 v2, 8, v9
.Ltmp10998:
	v_lshl_add_u32 v10, v2, 12, v7
.Ltmp10999:
	s_and_saveexec_b32 s1, vcc_lo
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execz .LBB19_7
.Ltmp11000:
	v_mov_b32_e32 v2, v1
.Ltmp11001:
	v_mov_b32_e32 v3, v1
	v_mov_b32_e32 v4, v1
	ds_write_b128 v10, v[1:4]
.Ltmp11002:
.LBB19_7:
	s_andn2_saveexec_b32 s10, s1
	s_cbranch_execz .LBB19_4
.Ltmp11003:
	v_add_lshl_u32 v2, v2, v5, 8
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[2:3], 1, v[2:3]
	v_add_co_u32 v2, s1, v6, v2
	v_add_co_ci_u32_e64 v3, null, v8, v3, s1
.Ltmp11004:
	global_load_dwordx4 v[11:14], v[2:3], off
	s_waitcnt vmcnt(0)
	ds_write_b128 v10, v[11:14]
.Ltmp11005:
	s_branch .LBB19_4
.Ltmp11006:
.LBB19_9:
	s_inst_prefetch 0x2
	s_or_b32 exec_lo, exec_lo, s3
	s_sub_i32 s3, s41, s9
	s_mov_b32 s21, 0
.Ltmp11007:
	s_add_i32 s3, s3, s2
.Ltmp11008:
	s_cmp_gt_i32 s31, 0
	s_mov_b32 s43, 0
	s_cselect_b32 s20, -1, 0
.Ltmp11009:
	s_waitcnt lgkmcnt(0)
.Ltmp11010:
	s_and_b32 vcc_lo, exec_lo, s20
.Ltmp11011:
	s_barrier
	buffer_gl0_inv
	s_cbranch_vccz .LBB19_12
.Ltmp11012:
	s_sub_i32 s1, s3, s31
.Ltmp11013:
	s_cmp_lt_i32 s1, 0
	s_cbranch_scc1 .LBB19_12
.Ltmp11014:
	s_add_i32 s1, s1, 1
.Ltmp11015:
	s_and_b32 s43, s1, -16
.Ltmp11016:
.LBB19_12:
	s_add_i32 s1, s3, s33
	v_and_b32_e32 v33, 15, v0
.Ltmp11017:
	v_bfe_u32 v35, v0, 4, 3
.Ltmp11018:
	s_min_i32 s1, s2, s1
.Ltmp11019:
	s_cmp_eq_u32 s30, 0
	s_cselect_b32 s44, s2, s1
.Ltmp11020:
	v_lshlrev_b32_e32 v34, 4, v33
.Ltmp11021:
	v_cmp_gt_i32_e64 s1, s42, v35
.Ltmp11022:
	s_cmp_ge_i32 s43, s44
	s_cbranch_scc1 .LBB19_102
.Ltmp11023:
	s_clause 0x2
	s_load_dwordx4 s[36:39], s[4:5], 0x50
	s_load_dwordx8 s[12:19], s[4:5], 0x30
	s_load_dword s4, s[4:5], 0x60
.Ltmp11024:
	v_mbcnt_lo_u32_b32 v17, -1, 0
.Ltmp11025:
	v_lshlrev_b32_e32 v4, 3, v0
	v_add_nc_u32_e32 v37, s3, v35
	s_movk_i32 s3, 0x210
	v_bfe_u32 v18, v0, 1, 3
	v_bfi_b32 v19, v17, 0, 32
	v_xor_b32_e32 v21, 8, v17
	v_xor_b32_e32 v22, 4, v17
	v_xor_b32_e32 v24, 1, v17
	v_and_b32_e32 v48, 8, v4
	v_xor_b32_e32 v23, 2, v17
	v_cmp_lt_u32_e32 vcc_lo, v21, v19
	v_mad_u32_u24 v43, v33, s3, 0x80
.Ltmp11026:
	v_lshrrev_b32_e32 v20, 4, v0
.Ltmp11027:
	v_mad_u32_u24 v58, v48, s3, 0x80
	v_cmp_lt_u32_e64 s3, v23, v19
	v_cndmask_b32_e32 v21, v17, v21, vcc_lo
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s8, s38, s8
.Ltmp11028:
	v_cmp_lt_u32_e32 vcc_lo, v22, v19
	s_ashr_i32 s9, s8, 31
.Ltmp11029:
	v_cndmask_b32_e64 v23, v17, v23, s3
	s_lshl_b64 s[8:9], s[8:9], 2
	v_mul_lo_u32 v18, v18, s37
	s_add_u32 s26, s26, s8
	s_addc_u32 s27, s27, s9
	s_cmp_lg_u32 s16, 1
	s_mul_i32 s8, s6, s13
	s_cselect_b32 s2, -1, 0
.Ltmp11030:
	s_cmp_lg_u32 s36, 1
	s_mul_i32 s6, s6, s18
.Ltmp11031:
	s_cselect_b32 s5, -1, 0
	v_cndmask_b32_e32 v22, v17, v22, vcc_lo
	s_or_b32 s2, s2, s5
	s_cmp_lg_u32 s4, 8
	v_lshlrev_b32_e32 v60, 2, v21
	s_cselect_b32 s5, -1, 0
	v_lshlrev_b32_e32 v61, 2, v22
	s_or_b32 s2, s2, s5
	s_and_b32 s5, s39, 7
	s_cselect_b32 s5, -1, 0
	s_ashr_i32 s9, s8, 31
	s_or_b32 s13, s5, s2
	s_lshl_b64 s[8:9], s[8:9], 1
	v_lshlrev_b32_e32 v62, 2, v23
	s_add_u32 s18, s22, s8
	s_addc_u32 s22, s23, s9
	s_ashr_i32 s7, s6, 31
	v_lshl_add_u32 v44, v20, 9, 0x80
	s_lshl_b64 s[6:7], s[6:7], 1
.Ltmp11032:
	v_mul_lo_u32 v64, s14, v20
	s_add_u32 s23, s24, s6
	s_addc_u32 s24, s25, s7
	s_cmp_lg_u32 s30, 0
	v_and_b32_e32 v3, 0xf0, v0
	s_cselect_b32 s25, -1, 0
	s_abs_i32 s5, s4
	s_abs_i32 s30, s39
.Ltmp11033:
	v_cvt_f32_u32_e32 v1, s5
	s_sub_i32 s6, 0, s5
	v_cvt_f32_u32_e32 v27, s30
	s_ashr_i32 s3, s4, 31
	v_lshlrev_b32_e32 v36, 2, v0
	v_rcp_iflag_f32_e32 v2, v1
	v_mov_b32_e32 v1, 0
	v_rcp_iflag_f32_e32 v27, v27
	v_lshlrev_b32_e32 v49, 2, v48
	v_lshl_add_u32 v47, v3, 2, 0x80
	v_cmp_gt_u32_e64 s2, 16, v0
	v_mov_b32_e32 v3, v1
	v_mov_b32_e32 v4, v1
	v_mov_b32_e32 v6, v1
	v_mov_b32_e32 v7, v1
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_mov_b32_e32 v8, v1
	v_mov_b32_e32 v9, v1
	v_mov_b32_e32 v10, v1
	v_mov_b32_e32 v11, v1
	v_cvt_u32_f32_e32 v12, v2
	v_mov_b32_e32 v2, v1
	v_mov_b32_e32 v14, v1
	v_mov_b32_e32 v15, v1
	v_lshl_add_u32 v38, v34, 1, 0x80
	v_mul_lo_u32 v5, s6, v12
	s_sub_i32 s6, 0, s30
	v_lshlrev_b32_e32 v39, 2, v33
	v_lshl_add_u32 v40, v0, 1, 0x80
	v_or_b32_e32 v41, 0x100, v0
	v_mov_b32_e32 v42, 0xff800000
	v_mov_b32_e32 v45, 0
	v_add_nc_u32_e32 v46, 0x80, v36
	v_mul_hi_u32 v13, v12, v5
	v_mov_b32_e32 v5, v1
	v_or_b32_e32 v50, 1, v48
	v_or_b32_e32 v51, 2, v48
	v_or_b32_e32 v52, 3, v48
	v_or_b32_e32 v53, 4, v48
	v_mul_u32_u24_e32 v54, 0x210, v48
	v_or_b32_e32 v55, 5, v48
	v_add_nc_u32_e32 v16, v12, v13
	v_mov_b32_e32 v12, v1
	v_mov_b32_e32 v13, v1
	v_or_b32_e32 v56, 6, v48
	v_or_b32_e32 v57, 7, v48
	v_mul_hi_u32 v25, v0, v16
	v_mov_b32_e32 v16, v1
	v_or_b32_e32 v59, 64, v49
	v_mul_lo_u32 v26, v25, s5
	v_add_nc_u32_e32 v28, 1, v25
	v_sub_nc_u32_e32 v26, v0, v26
	v_subrev_nc_u32_e32 v29, s5, v26
	v_cmp_le_u32_e32 vcc_lo, s5, v26
	v_cndmask_b32_e32 v25, v25, v28, vcc_lo
	v_cndmask_b32_e32 v26, v26, v29, vcc_lo
	v_cmp_lt_u32_e32 vcc_lo, v24, v19
	v_add_nc_u32_e32 v28, 1, v25
	v_subrev_nc_u32_e32 v21, s5, v26
	v_cndmask_b32_e32 v24, v17, v24, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s5, v26
	v_mov_b32_e32 v17, v16
	v_mov_b32_e32 v16, v15
	v_mov_b32_e32 v15, v14
	v_lshlrev_b32_e32 v63, 2, v24
	v_cndmask_b32_e32 v19, v25, v28, vcc_lo
	v_mul_f32_e32 v25, 0x4f7ffffe, v27
	v_cndmask_b32_e32 v23, v26, v21, vcc_lo
	v_mov_b32_e32 v14, v13
	v_mov_b32_e32 v13, v12
	v_xor_b32_e32 v19, s3, v19
	v_cvt_u32_f32_e32 v29, v25
	v_mul_lo_u32 v21, v23, s16
	v_mul_lo_u32 v23, v23, s37
	v_mov_b32_e32 v12, v11
	v_subrev_nc_u32_e32 v22, s3, v19
	v_mul_lo_u32 v25, s6, v29
	v_ashrrev_i32_e32 v19, 31, v18
	v_mov_b32_e32 v11, v10
	v_mov_b32_e32 v10, v9
	v_mul_lo_u32 v27, v22, s4
	v_mul_lo_u32 v24, v22, s14
	v_mul_lo_u32 v26, v22, s19
	v_lshlrev_b64 v[19:20], 1, v[18:19]
.Ltmp11034:
	v_mul_hi_u32 v18, v29, v25
	v_mov_b32_e32 v9, v8
	v_mov_b32_e32 v8, v7
	v_mov_b32_e32 v7, v6
	v_sub_nc_u32_e32 v22, v0, v27
	v_ashrrev_i32_e32 v25, 31, v24
	v_ashrrev_i32_e32 v27, 31, v26
	v_add_co_u32 v65, vcc_lo, s23, v19
	v_mul_lo_u32 v28, v22, s16
	v_mul_lo_u32 v30, v22, s37
	v_lshlrev_b64 v[68:69], 1, v[24:25]
	v_lshlrev_b64 v[70:71], 1, v[26:27]
	v_add_nc_u32_e32 v67, v29, v18
	v_add_co_ci_u32_e64 v66, null, s24, v20, vcc_lo
	v_mov_b32_e32 v6, v5
	v_ashrrev_i32_e32 v29, 31, v28
	v_ashrrev_i32_e32 v31, 31, v30
	v_add_co_u32 v68, vcc_lo, s18, v68
	v_add_co_ci_u32_e64 v69, null, s22, v69, vcc_lo
	v_add_co_u32 v70, vcc_lo, s23, v70
	v_lshlrev_b64 v[25:26], 1, v[28:29]
	v_lshlrev_b64 v[27:28], 1, v[30:31]
	v_mov_b32_e32 v5, v4
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v2, v1
	v_ashrrev_i32_e32 v22, 31, v21
	v_ashrrev_i32_e32 v24, 31, v23
	v_add_co_ci_u32_e64 v71, null, s24, v71, vcc_lo
	s_ashr_i32 s16, s39, 31
	s_lshl_b32 s14, s14, 4
	s_branch .LBB19_15
.Ltmp11035:
.LBB19_14:
	s_or_b32 exec_lo, exec_lo, s3
.Ltmp11036:
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v18, v18, v30
.Ltmp11037:
	s_add_i32 s43, s43, 16
	s_cmp_ge_i32 s43, s44
.Ltmp11038:
	s_barrier
.Ltmp11039:
	v_fmac_f32_e32 v18, v45, v29
.Ltmp11040:
	buffer_gl0_inv
	v_mov_b32_e32 v45, v18
.Ltmp11041:
	s_cbranch_scc1 .LBB19_103
.Ltmp11042:
.LBB19_15:
	s_sub_i32 s37, s44, s43
.Ltmp11043:
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB19_21
.Ltmp11044:
	s_mov_b32 s4, exec_lo
	v_cmpx_le_i32_e64 s37, v0
	s_xor_b32 s4, exec_lo, s4
.Ltmp11045:
	ds_write_b32 v36, v1
.Ltmp11046:
	s_or_saveexec_b32 s4, s4
	v_mov_b32_e32 v18, 0
	s_xor_b32 exec_lo, exec_lo, s4
	s_cbranch_execz .LBB19_20
.Ltmp11047:
	v_or_b32_e32 v18, s43, v0
.Ltmp11048:
	v_sub_nc_u32_e32 v29, 0, v18
	v_max_i32_e32 v29, v29, v18
	v_mul_hi_u32 v30, v29, v67
	v_mul_lo_u32 v31, v30, s30
	v_sub_nc_u32_e32 v29, v29, v31
	v_add_nc_u32_e32 v31, 1, v30
	v_subrev_nc_u32_e32 v32, s30, v29
	v_cmp_le_u32_e32 vcc_lo, s30, v29
	v_cndmask_b32_e32 v30, v30, v31, vcc_lo
	v_cndmask_b32_e32 v29, v29, v32, vcc_lo
	v_ashrrev_i32_e32 v31, 31, v18
	v_add_nc_u32_e32 v32, 1, v30
	v_cmp_le_u32_e32 vcc_lo, s30, v29
	v_xor_b32_e32 v31, s16, v31
	v_cndmask_b32_e32 v29, v30, v32, vcc_lo
	v_xor_b32_e32 v29, v29, v31
	v_sub_nc_u32_e32 v29, v29, v31
	v_ashrrev_i32_e32 v30, 31, v29
	v_lshlrev_b64 v[30:31], 2, v[29:30]
	v_mul_lo_u32 v29, v29, s39
	v_add_co_u32 v30, vcc_lo, s26, v30
	v_add_co_ci_u32_e64 v31, null, s27, v31, vcc_lo
	v_sub_nc_u32_e32 v18, v18, v29
.Ltmp11049:
	global_load_dword v30, v[30:31], off
	s_waitcnt vmcnt(0)
	ds_write_b32 v36, v30
.Ltmp11050:
.LBB19_20:
	s_or_b32 exec_lo, exec_lo, s4
	ds_write_b32 v36, v18 offset:64
.Ltmp11051:
.LBB19_21:
	s_or_b32 exec_lo, exec_lo, s3
.Ltmp11052:
	s_min_i32 s9, s37, 16
.Ltmp11053:
	s_andn2_b32 vcc_lo, exec_lo, s13
	s_mov_b32 s3, -1
.Ltmp11054:
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
.Ltmp11055:
	s_cbranch_vccnz .LBB19_33
.Ltmp11056:
	v_mov_b32_e32 v30, v0
.Ltmp11057:
	s_and_saveexec_b32 s3, s21
	s_cbranch_execz .LBB19_26
.Ltmp11058:
	s_cmp_gt_i32 s9, 0
	s_cselect_b32 s5, -1, 0
	s_and_saveexec_b32 s4, s5
	s_cbranch_execz .LBB19_25
.Ltmp11059:
	ds_read2_b32 v[29:30], v1 offset1:16
	v_lshlrev_b64 v[76:77], 1, v[21:22]
	v_lshlrev_b64 v[78:79], 1, v[23:24]
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v31, v29, s12
	v_mul_lo_u32 v29, v29, s17
	v_mul_lo_u32 v72, v30, s15
.Ltmp11060:
	v_mul_lo_u32 v74, v30, s36
	v_ashrrev_i32_e32 v32, 31, v31
	v_ashrrev_i32_e32 v30, 31, v29
	v_ashrrev_i32_e32 v73, 31, v72
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshlrev_b64 v[31:32], 1, v[31:32]
	v_lshlrev_b64 v[29:30], 1, v[29:30]
	v_lshlrev_b64 v[72:73], 1, v[72:73]
	v_lshlrev_b64 v[74:75], 1, v[74:75]
	v_add_co_u32 v18, vcc_lo, v68, v31
	v_add_co_ci_u32_e64 v31, null, v69, v32, vcc_lo
	v_add_co_u32 v29, vcc_lo, v70, v29
	v_add_co_ci_u32_e64 v30, null, v71, v30, vcc_lo
	v_add_co_u32 v18, vcc_lo, v18, v72
	v_add_co_ci_u32_e64 v31, null, v31, v73, vcc_lo
	v_add_co_u32 v32, vcc_lo, v29, v74
	v_add_co_ci_u32_e64 v72, null, v30, v75, vcc_lo
	v_add_co_u32 v29, vcc_lo, v18, v76
	v_add_co_ci_u32_e64 v30, null, v31, v77, vcc_lo
	v_add_co_u32 v31, vcc_lo, v32, v78
	v_add_co_ci_u32_e64 v32, null, v72, v79, vcc_lo
.Ltmp11061:
	global_load_ushort v18, v[29:30], off
.Ltmp11062:
	global_load_ushort v29, v[31:32], off
.Ltmp11063:
	s_waitcnt vmcnt(1)
	ds_write_b16 v40, v18 offset:8192
	s_waitcnt vmcnt(0)
	ds_write_b16 v40, v29 offset:16640
.Ltmp11064:
.LBB19_25:
	s_or_b32 exec_lo, exec_lo, s4
	v_mov_b32_e32 v30, v41
.Ltmp11065:
.LBB19_26:
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v18, 0x100, v30
	v_lshrrev_b32_e32 v29, 8, v30
	v_or_b32_e32 v30, 0xfffffe00, v30
	s_mov_b32 s3, 0
	v_lshrrev_b32_e32 v18, 8, v18
	v_lshlrev_b32_e32 v32, 2, v29
	v_lshlrev_b32_e32 v31, 2, v18
	s_branch .LBB19_28
.Ltmp11066:
.LBB19_27:
	s_or_b32 exec_lo, exec_lo, s4
.Ltmp11067:
	v_add_nc_u32_e32 v30, 0x200, v30
.Ltmp11068:
	v_add_nc_u32_e32 v18, 2, v18
	v_add_nc_u32_e32 v31, 8, v31
	v_add_nc_u32_e32 v29, 2, v29
	v_add_nc_u32_e32 v32, 8, v32
	v_cmp_lt_u32_e32 vcc_lo, 0xdff, v30
.Ltmp11069:
	s_or_b32 s3, vcc_lo, s3
	s_andn2_b32 exec_lo, exec_lo, s3
	s_cbranch_execz .LBB19_32
.Ltmp11070:
.LBB19_28:
	s_mov_b32 s4, exec_lo
	v_cmpx_gt_i32_e64 s9, v29
	s_cbranch_execz .LBB19_30
.Ltmp11071:
	ds_read2_b32 v[72:73], v32 offset1:16
	v_lshlrev_b64 v[80:81], 1, v[21:22]
	v_lshlrev_b64 v[82:83], 1, v[23:24]
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v74, v72, s12
	v_mul_lo_u32 v72, v72, s17
	v_mul_lo_u32 v76, v73, s15
.Ltmp11072:
	v_mul_lo_u32 v78, v73, s36
	v_ashrrev_i32_e32 v75, 31, v74
	v_ashrrev_i32_e32 v73, 31, v72
	v_ashrrev_i32_e32 v77, 31, v76
	v_ashrrev_i32_e32 v79, 31, v78
	v_lshlrev_b64 v[74:75], 1, v[74:75]
	v_lshlrev_b64 v[72:73], 1, v[72:73]
	v_lshlrev_b64 v[76:77], 1, v[76:77]
	v_lshlrev_b64 v[78:79], 1, v[78:79]
	v_add_co_u32 v74, vcc_lo, v68, v74
	v_add_co_ci_u32_e64 v75, null, v69, v75, vcc_lo
	v_add_co_u32 v72, vcc_lo, v70, v72
	v_add_co_ci_u32_e64 v73, null, v71, v73, vcc_lo
	v_add_co_u32 v74, vcc_lo, v74, v76
	v_add_co_ci_u32_e64 v75, null, v75, v77, vcc_lo
	v_add_co_u32 v76, vcc_lo, v72, v78
	v_add_co_ci_u32_e64 v77, null, v73, v79, vcc_lo
	v_add_co_u32 v72, vcc_lo, v74, v80
	v_add_co_ci_u32_e64 v73, null, v75, v81, vcc_lo
	v_add_co_u32 v74, vcc_lo, v76, v82
	v_add_co_ci_u32_e64 v75, null, v77, v83, vcc_lo
.Ltmp11073:
	global_load_ushort v72, v[72:73], off
.Ltmp11074:
	global_load_ushort v73, v[74:75], off
.Ltmp11075:
	v_mad_u32_u24 v74, 0x108, v29, v0
	v_lshl_add_u32 v74, v74, 1, 0x80
	s_waitcnt vmcnt(1)
	ds_write_b16 v74, v72 offset:8192
	s_waitcnt vmcnt(0)
	ds_write_b16 v74, v73 offset:16640
.Ltmp11076:
.LBB19_30:
	s_or_b32 exec_lo, exec_lo, s4
.Ltmp11077:
	s_mov_b32 s4, exec_lo
	v_cmpx_gt_i32_e64 s9, v18
	s_cbranch_execz .LBB19_27
.Ltmp11078:
	ds_read2_b32 v[72:73], v31 offset1:16
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v74, v72, s12
	v_mul_lo_u32 v72, v72, s17
	v_mul_lo_u32 v76, v73, s15
.Ltmp11079:
	v_mul_lo_u32 v78, v73, s36
	v_ashrrev_i32_e32 v75, 31, v74
	v_ashrrev_i32_e32 v73, 31, v72
	v_ashrrev_i32_e32 v77, 31, v76
	v_ashrrev_i32_e32 v79, 31, v78
	v_lshlrev_b64 v[74:75], 1, v[74:75]
	v_lshlrev_b64 v[72:73], 1, v[72:73]
	v_lshlrev_b64 v[76:77], 1, v[76:77]
	v_lshlrev_b64 v[78:79], 1, v[78:79]
	v_add_co_u32 v74, vcc_lo, v68, v74
	v_add_co_ci_u32_e64 v75, null, v69, v75, vcc_lo
	v_add_co_u32 v72, vcc_lo, v70, v72
	v_add_co_ci_u32_e64 v73, null, v71, v73, vcc_lo
	v_add_co_u32 v74, vcc_lo, v74, v76
	v_add_co_ci_u32_e64 v75, null, v75, v77, vcc_lo
	v_add_co_u32 v76, vcc_lo, v72, v78
	v_add_co_ci_u32_e64 v77, null, v73, v79, vcc_lo
	v_add_co_u32 v72, vcc_lo, v74, v25
	v_add_co_ci_u32_e64 v73, null, v75, v26, vcc_lo
	v_add_co_u32 v74, vcc_lo, v76, v27
	v_add_co_ci_u32_e64 v75, null, v77, v28, vcc_lo
.Ltmp11080:
	global_load_ushort v72, v[72:73], off
.Ltmp11081:
	global_load_ushort v73, v[74:75], off
.Ltmp11082:
	v_mad_u32_u24 v74, 0x108, v18, v0
	v_lshl_add_u32 v74, v74, 1, 0x80
	s_waitcnt vmcnt(1)
	ds_write_b16 v74, v72 offset:8192
	s_waitcnt vmcnt(0)
	ds_write_b16 v74, v73 offset:16640
.Ltmp11083:
	s_branch .LBB19_27
.Ltmp11084:
.LBB19_32:
	s_or_b32 exec_lo, exec_lo, s3
	s_mov_b32 s3, 0
.Ltmp11085:
.LBB19_33:
	s_and_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccz .LBB19_61
.Ltmp11086:
	s_and_saveexec_b32 s38, s0
	s_cbranch_execz .LBB19_60
.Ltmp11087:
	v_mov_b32_e32 v29, v64
	v_mov_b32_e32 v18, v0
	s_mov_b32 s4, 0
	v_cmp_gt_i32_e32 vcc_lo, s37, v33
	s_inst_prefetch 0x1
	s_branch .LBB19_37
.Ltmp11088:
	.p2align	6
.LBB19_36:
	s_or_b32 exec_lo, exec_lo, s5
.Ltmp11089:
	v_add_nc_u32_e32 v30, 0x100, v18
.Ltmp11090:
	v_cmp_lt_u32_e64 s3, 0xff, v18
	v_add_nc_u32_e32 v29, s14, v29
	v_mov_b32_e32 v18, v30
.Ltmp11091:
	s_or_b32 s4, s3, s4
	s_andn2_b32 exec_lo, exec_lo, s4
	s_cbranch_execz .LBB19_39
.Ltmp11092:
.LBB19_37:
	s_and_saveexec_b32 s5, vcc_lo
	s_cbranch_execz .LBB19_36
.Ltmp11093:
	ds_read2_b32 v[30:31], v39 offset1:16
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v72, v30, s12
	v_mul_lo_u32 v31, v31, s15
	v_ashrrev_i32_e32 v30, 31, v29
	v_lshlrev_b64 v[74:75], 1, v[29:30]
	v_ashrrev_i32_e32 v73, 31, v72
	v_ashrrev_i32_e32 v32, 31, v31
	v_lshlrev_b64 v[72:73], 1, v[72:73]
	v_lshlrev_b64 v[30:31], 1, v[31:32]
	v_add_co_u32 v72, s3, s18, v72
	v_add_co_ci_u32_e64 v73, null, s22, v73, s3
	v_add_co_u32 v32, s3, v72, v74
	v_add_co_ci_u32_e64 v72, null, v73, v75, s3
	v_add_co_u32 v30, s3, v32, v30
	v_add_co_ci_u32_e64 v31, null, v72, v31, s3
.Ltmp11094:
	global_load_dwordx4 v[72:75], v[30:31], off
	v_and_b32_e32 v30, 0x1f0, v18
	v_add_nc_u32_e32 v30, v43, v30
	s_waitcnt vmcnt(0)
	ds_write_b128 v30, v[72:75] offset:8192
	s_branch .LBB19_36
.Ltmp11095:
.LBB19_39:
	s_inst_prefetch 0x2
	s_or_b32 exec_lo, exec_lo, s4
	v_cmp_gt_i32_e64 s3, s9, v50
	v_cmp_gt_i32_e64 s4, s9, v51
	v_cmp_gt_i32_e64 s5, s9, v52
	v_cmp_gt_i32_e64 s6, s9, v53
	v_cmp_gt_i32_e64 s7, s9, v55
	v_cmp_gt_i32_e64 s8, s9, v56
	v_cmp_gt_i32_e64 s9, s9, v57
.Ltmp11096:
	v_mov_b32_e32 v18, v0
	s_mov_b32 s45, 0
	v_cmp_gt_i32_e32 vcc_lo, s37, v48
	s_branch .LBB19_41
.Ltmp11097:
.LBB19_40:
	s_or_b32 exec_lo, exec_lo, s46
.Ltmp11098:
	v_add_nc_u32_e32 v29, 0x100, v18
.Ltmp11099:
	v_cmp_lt_u32_e64 s10, 0xff, v18
	v_mov_b32_e32 v18, v29
.Ltmp11100:
	s_or_b32 s45, s10, s45
	s_andn2_b32 exec_lo, exec_lo, s45
	s_cbranch_execz .LBB19_60
.Ltmp11101:
.LBB19_41:
	s_and_saveexec_b32 s46, vcc_lo
	s_cbranch_execz .LBB19_40
.Ltmp11102:
	ds_read_b32 v29, v59
.Ltmp11103:
	v_lshrrev_b32_e32 v30, 4, v18
.Ltmp11104:
	v_lshrrev_b32_e32 v72, 1, v18
.Ltmp11105:
	s_mov_b32 s47, exec_lo
.Ltmp11106:
	v_mul_lo_u32 v30, v30, s19
	v_ashrrev_i32_e32 v31, 31, v30
	v_lshlrev_b64 v[31:32], 1, v[30:31]
.Ltmp11107:
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v73, 7, v29
	v_ashrrev_i32_e32 v30, 31, v29
	v_cmpx_ne_u32_e32 0, v73
	s_xor_b32 s47, exec_lo, s47
	s_cbranch_execz .LBB19_58
.Ltmp11108:
	ds_read_b32 v73, v49
.Ltmp11109:
	v_add_co_u32 v31, s10, v65, v31
	v_add_co_ci_u32_e64 v32, null, v66, v32, s10
.Ltmp11110:
	v_lshlrev_b64 v[29:30], 1, v[29:30]
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v73, v73, s17
	v_ashrrev_i32_e32 v74, 31, v73
	v_lshlrev_b64 v[73:74], 1, v[73:74]
	v_add_co_u32 v73, s10, v31, v73
	v_add_co_ci_u32_e64 v74, null, v32, v74, s10
	v_add_co_u32 v29, s10, v73, v29
	v_add_co_ci_u32_e64 v30, null, v74, v30, s10
	global_load_ushort v30, v[29:30], off
.Ltmp11111:
	v_lshl_add_u32 v29, v72, 1, 0x80
.Ltmp11112:
	v_add_nc_u32_e32 v29, v29, v54
	s_waitcnt vmcnt(0)
	ds_write_b16 v29, v30 offset:16640
.Ltmp11113:
	s_and_saveexec_b32 s48, s3
	s_cbranch_execz .LBB19_45
.Ltmp11114:
	ds_read2_b32 v[72:73], v49 offset0:1 offset1:17
.Ltmp11115:
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v75, v72, s17
	v_ashrrev_i32_e32 v74, 31, v73
	v_lshlrev_b64 v[72:73], 1, v[73:74]
	v_ashrrev_i32_e32 v76, 31, v75
	v_lshlrev_b64 v[75:76], 1, v[75:76]
	v_add_co_u32 v30, s10, v31, v75
	v_add_co_ci_u32_e64 v74, null, v32, v76, s10
	v_add_co_u32 v72, s10, v30, v72
	v_add_co_ci_u32_e64 v73, null, v74, v73, s10
	global_load_ushort v30, v[72:73], off
	s_waitcnt vmcnt(0)
	ds_write_b16 v29, v30 offset:17168
.Ltmp11116:
.LBB19_45:
	s_or_b32 exec_lo, exec_lo, s48
.Ltmp11117:
	s_and_saveexec_b32 s48, s4
	s_cbranch_execz .LBB19_47
.Ltmp11118:
	ds_read2_b32 v[72:73], v49 offset0:2 offset1:18
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v75, v72, s17
	v_ashrrev_i32_e32 v74, 31, v73
	v_lshlrev_b64 v[72:73], 1, v[73:74]
	v_ashrrev_i32_e32 v76, 31, v75
	v_lshlrev_b64 v[75:76], 1, v[75:76]
	v_add_co_u32 v30, s10, v31, v75
	v_add_co_ci_u32_e64 v74, null, v32, v76, s10
	v_add_co_u32 v72, s10, v30, v72
	v_add_co_ci_u32_e64 v73, null, v74, v73, s10
	global_load_ushort v30, v[72:73], off
	s_waitcnt vmcnt(0)
	ds_write_b16 v29, v30 offset:17696
.Ltmp11119:
.LBB19_47:
	s_or_b32 exec_lo, exec_lo, s48
.Ltmp11120:
	s_and_saveexec_b32 s48, s5
	s_cbranch_execz .LBB19_49
.Ltmp11121:
	ds_read2_b32 v[72:73], v49 offset0:3 offset1:19
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v75, v72, s17
	v_ashrrev_i32_e32 v74, 31, v73
	v_lshlrev_b64 v[72:73], 1, v[73:74]
	v_ashrrev_i32_e32 v76, 31, v75
	v_lshlrev_b64 v[75:76], 1, v[75:76]
	v_add_co_u32 v30, s10, v31, v75
	v_add_co_ci_u32_e64 v74, null, v32, v76, s10
	v_add_co_u32 v72, s10, v30, v72
	v_add_co_ci_u32_e64 v73, null, v74, v73, s10
	global_load_ushort v30, v[72:73], off
	s_waitcnt vmcnt(0)
	ds_write_b16 v29, v30 offset:18224
.Ltmp11122:
.LBB19_49:
	s_or_b32 exec_lo, exec_lo, s48
.Ltmp11123:
	s_and_saveexec_b32 s48, s6
	s_cbranch_execz .LBB19_51
.Ltmp11124:
	ds_read2_b32 v[72:73], v49 offset0:4 offset1:20
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v75, v72, s17
	v_ashrrev_i32_e32 v74, 31, v73
	v_lshlrev_b64 v[72:73], 1, v[73:74]
	v_ashrrev_i32_e32 v76, 31, v75
	v_lshlrev_b64 v[75:76], 1, v[75:76]
	v_add_co_u32 v30, s10, v31, v75
	v_add_co_ci_u32_e64 v74, null, v32, v76, s10
	v_add_co_u32 v72, s10, v30, v72
	v_add_co_ci_u32_e64 v73, null, v74, v73, s10
	global_load_ushort v30, v[72:73], off
	s_waitcnt vmcnt(0)
	ds_write_b16 v29, v30 offset:18752
.Ltmp11125:
.LBB19_51:
	s_or_b32 exec_lo, exec_lo, s48
.Ltmp11126:
	s_and_saveexec_b32 s48, s7
	s_cbranch_execz .LBB19_53
.Ltmp11127:
	ds_read2_b32 v[72:73], v49 offset0:5 offset1:21
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v75, v72, s17
	v_ashrrev_i32_e32 v74, 31, v73
	v_lshlrev_b64 v[72:73], 1, v[73:74]
	v_ashrrev_i32_e32 v76, 31, v75
	v_lshlrev_b64 v[75:76], 1, v[75:76]
	v_add_co_u32 v30, s10, v31, v75
	v_add_co_ci_u32_e64 v74, null, v32, v76, s10
	v_add_co_u32 v72, s10, v30, v72
	v_add_co_ci_u32_e64 v73, null, v74, v73, s10
	global_load_ushort v30, v[72:73], off
	s_waitcnt vmcnt(0)
	ds_write_b16 v29, v30 offset:19280
.Ltmp11128:
.LBB19_53:
	s_or_b32 exec_lo, exec_lo, s48
.Ltmp11129:
	s_and_saveexec_b32 s48, s8
	s_cbranch_execz .LBB19_55
.Ltmp11130:
	ds_read2_b32 v[72:73], v49 offset0:6 offset1:22
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v75, v72, s17
	v_ashrrev_i32_e32 v74, 31, v73
	v_lshlrev_b64 v[72:73], 1, v[73:74]
	v_ashrrev_i32_e32 v76, 31, v75
	v_lshlrev_b64 v[75:76], 1, v[75:76]
	v_add_co_u32 v30, s10, v31, v75
	v_add_co_ci_u32_e64 v74, null, v32, v76, s10
	v_add_co_u32 v72, s10, v30, v72
	v_add_co_ci_u32_e64 v73, null, v74, v73, s10
	global_load_ushort v30, v[72:73], off
	s_waitcnt vmcnt(0)
	ds_write_b16 v29, v30 offset:19808
.Ltmp11131:
.LBB19_55:
	s_or_b32 exec_lo, exec_lo, s48
.Ltmp11132:
	s_and_saveexec_b32 s48, s9
	s_cbranch_execz .LBB19_57
.Ltmp11133:
	ds_read2_b32 v[72:73], v49 offset0:7 offset1:23
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v75, v72, s17
	v_ashrrev_i32_e32 v74, 31, v73
	v_lshlrev_b64 v[72:73], 1, v[73:74]
	v_ashrrev_i32_e32 v76, 31, v75
	v_lshlrev_b64 v[75:76], 1, v[75:76]
	v_add_co_u32 v30, s10, v31, v75
	v_add_co_ci_u32_e64 v31, null, v32, v76, s10
	v_add_co_u32 v30, s10, v30, v72
	v_add_co_ci_u32_e64 v31, null, v31, v73, s10
	global_load_ushort v30, v[30:31], off
	s_waitcnt vmcnt(0)
	ds_write_b16 v29, v30 offset:20336
.Ltmp11134:
.LBB19_57:
	s_or_b32 exec_lo, exec_lo, s48
.Ltmp11135:
.LBB19_58:
	s_andn2_saveexec_b32 s10, s47
	s_cbranch_execz .LBB19_40
.Ltmp11136:
	ds_read_b32 v73, v49
.Ltmp11137:
	v_lshlrev_b64 v[29:30], 1, v[29:30]
.Ltmp11138:
	v_lshl_add_u32 v72, v72, 1, v58
.Ltmp11139:
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v73, v73, s17
	v_ashrrev_i32_e32 v74, 31, v73
	v_lshlrev_b64 v[73:74], 1, v[73:74]
	v_add_co_u32 v73, s10, s23, v73
	v_add_co_ci_u32_e64 v74, null, s24, v74, s10
.Ltmp11140:
	v_add_co_u32 v31, s10, v73, v31
	v_add_co_ci_u32_e64 v32, null, v74, v32, s10
.Ltmp11141:
	v_add_co_u32 v31, s10, v31, v19
	v_add_co_ci_u32_e64 v32, null, v32, v20, s10
	v_add_co_u32 v29, s10, v31, v29
	v_add_co_ci_u32_e64 v30, null, v32, v30, s10
.Ltmp11142:
	global_load_dwordx4 v[29:32], v[29:30], off
.Ltmp11143:
	s_waitcnt vmcnt(0)
	ds_write_b16 v72, v29 offset:16640
.Ltmp11144:
	ds_write_b16_d16_hi v72, v29 offset:17168
.Ltmp11145:
	ds_write_b16 v72, v30 offset:17696
.Ltmp11146:
	ds_write_b16_d16_hi v72, v30 offset:18224
.Ltmp11147:
	ds_write_b16 v72, v31 offset:18752
.Ltmp11148:
	ds_write_b16_d16_hi v72, v31 offset:19280
.Ltmp11149:
	ds_write_b16 v72, v32 offset:19808
.Ltmp11150:
	ds_write_b16_d16_hi v72, v32 offset:20336
.Ltmp11151:
	s_branch .LBB19_40
.Ltmp11152:
.LBB19_60:
	s_or_b32 exec_lo, exec_lo, s38
.Ltmp11153:
.LBB19_61:
	v_cmp_gt_i32_e32 vcc_lo, s37, v33
	v_mov_b32_e32 v18, 0xff800000
.Ltmp11154:
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
.Ltmp11155:
	s_and_b32 s3, s1, vcc_lo
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB19_65
.Ltmp11156:
	v_or_b32_e32 v18, s43, v33
.Ltmp11157:
	v_sub_nc_u32_e32 v29, v37, v18
	v_cmp_lt_i32_e32 vcc_lo, v37, v18
	v_mov_b32_e32 v18, 0xff800000
.Ltmp11158:
	v_cmp_le_i32_e64 s3, s31, v29
	s_and_b32 s5, s25, vcc_lo
	s_and_b32 s3, s20, s3
	s_nor_b32 s5, s5, s3
.Ltmp11159:
	s_and_saveexec_b32 s3, s5
	s_cbranch_execz .LBB19_64
.Ltmp11160:
	ds_read_b128 v[29:32], v43 offset:8192
.Ltmp11161:
	ds_read_b128 v[72:75], v44
.Ltmp11162:
	ds_read_b128 v[76:79], v44 offset:16
.Ltmp11163:
	ds_read_b128 v[80:83], v43 offset:8208
.Ltmp11164:
	v_mov_b32_e32 v18, 0
.Ltmp11165:
	v_mov_b32_e32 v92, 0
.Ltmp11166:
	ds_read_b128 v[84:87], v44 offset:32
.Ltmp11167:
	ds_read_b128 v[88:91], v43 offset:8224
.Ltmp11168:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v72, v29
.Ltmp11169:
	v_dot2c_f32_f16 v92, v73, v30
.Ltmp11170:
	v_dot2c_f32_f16 v18, v74, v31
.Ltmp11171:
	v_dot2c_f32_f16 v92, v75, v32
.Ltmp11172:
	ds_read_b128 v[29:32], v44 offset:48
.Ltmp11173:
	ds_read_b128 v[72:75], v43 offset:8240
.Ltmp11174:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v76, v80
.Ltmp11175:
	v_dot2c_f32_f16 v92, v77, v81
.Ltmp11176:
	v_dot2c_f32_f16 v18, v78, v82
.Ltmp11177:
	v_dot2c_f32_f16 v92, v79, v83
.Ltmp11178:
	ds_read_b128 v[76:79], v44 offset:64
.Ltmp11179:
	ds_read_b128 v[80:83], v43 offset:8256
.Ltmp11180:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v84, v88
.Ltmp11181:
	v_dot2c_f32_f16 v92, v85, v89
.Ltmp11182:
	v_dot2c_f32_f16 v18, v86, v90
.Ltmp11183:
	v_dot2c_f32_f16 v92, v87, v91
.Ltmp11184:
	ds_read_b128 v[84:87], v44 offset:80
.Ltmp11185:
	ds_read_b128 v[88:91], v43 offset:8272
.Ltmp11186:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v29, v72
.Ltmp11187:
	v_dot2c_f32_f16 v92, v30, v73
.Ltmp11188:
	v_dot2c_f32_f16 v18, v31, v74
.Ltmp11189:
	v_dot2c_f32_f16 v92, v32, v75
.Ltmp11190:
	ds_read_b128 v[29:32], v44 offset:96
.Ltmp11191:
	ds_read_b128 v[72:75], v43 offset:8288
.Ltmp11192:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v76, v80
.Ltmp11193:
	v_dot2c_f32_f16 v92, v77, v81
.Ltmp11194:
	v_dot2c_f32_f16 v18, v78, v82
.Ltmp11195:
	v_dot2c_f32_f16 v92, v79, v83
.Ltmp11196:
	ds_read_b128 v[76:79], v44 offset:112
.Ltmp11197:
	ds_read_b128 v[80:83], v43 offset:8304
.Ltmp11198:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v84, v88
.Ltmp11199:
	v_dot2c_f32_f16 v92, v85, v89
.Ltmp11200:
	v_dot2c_f32_f16 v18, v86, v90
.Ltmp11201:
	v_dot2c_f32_f16 v92, v87, v91
.Ltmp11202:
	ds_read_b128 v[84:87], v44 offset:128
.Ltmp11203:
	ds_read_b128 v[88:91], v43 offset:8320
.Ltmp11204:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v29, v72
.Ltmp11205:
	v_dot2c_f32_f16 v92, v30, v73
.Ltmp11206:
	v_dot2c_f32_f16 v18, v31, v74
.Ltmp11207:
	v_dot2c_f32_f16 v92, v32, v75
.Ltmp11208:
	ds_read_b128 v[29:32], v44 offset:144
.Ltmp11209:
	ds_read_b128 v[72:75], v43 offset:8336
.Ltmp11210:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v76, v80
.Ltmp11211:
	v_dot2c_f32_f16 v92, v77, v81
.Ltmp11212:
	v_dot2c_f32_f16 v18, v78, v82
.Ltmp11213:
	v_dot2c_f32_f16 v92, v79, v83
.Ltmp11214:
	ds_read_b128 v[76:79], v44 offset:160
.Ltmp11215:
	ds_read_b128 v[80:83], v43 offset:8352
.Ltmp11216:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v84, v88
.Ltmp11217:
	v_dot2c_f32_f16 v92, v85, v89
.Ltmp11218:
	v_dot2c_f32_f16 v18, v86, v90
.Ltmp11219:
	v_dot2c_f32_f16 v92, v87, v91
.Ltmp11220:
	ds_read_b128 v[84:87], v44 offset:176
.Ltmp11221:
	ds_read_b128 v[88:91], v43 offset:8368
.Ltmp11222:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v29, v72
.Ltmp11223:
	v_dot2c_f32_f16 v92, v30, v73
.Ltmp11224:
	v_dot2c_f32_f16 v18, v31, v74
.Ltmp11225:
	v_dot2c_f32_f16 v92, v32, v75
.Ltmp11226:
	ds_read_b128 v[29:32], v44 offset:192
.Ltmp11227:
	ds_read_b128 v[72:75], v43 offset:8384
.Ltmp11228:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v76, v80
.Ltmp11229:
	v_dot2c_f32_f16 v92, v77, v81
.Ltmp11230:
	v_dot2c_f32_f16 v18, v78, v82
.Ltmp11231:
	v_dot2c_f32_f16 v92, v79, v83
.Ltmp11232:
	ds_read_b128 v[76:79], v44 offset:208
.Ltmp11233:
	ds_read_b128 v[80:83], v43 offset:8400
.Ltmp11234:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v84, v88
.Ltmp11235:
	v_dot2c_f32_f16 v92, v85, v89
.Ltmp11236:
	v_dot2c_f32_f16 v18, v86, v90
.Ltmp11237:
	v_dot2c_f32_f16 v92, v87, v91
.Ltmp11238:
	ds_read_b128 v[84:87], v44 offset:224
.Ltmp11239:
	ds_read_b128 v[88:91], v43 offset:8416
.Ltmp11240:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v29, v72
.Ltmp11241:
	v_dot2c_f32_f16 v92, v30, v73
.Ltmp11242:
	v_dot2c_f32_f16 v18, v31, v74
.Ltmp11243:
	v_dot2c_f32_f16 v92, v32, v75
.Ltmp11244:
	ds_read_b128 v[29:32], v44 offset:240
.Ltmp11245:
	ds_read_b128 v[72:75], v43 offset:8432
.Ltmp11246:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v76, v80
.Ltmp11247:
	v_dot2c_f32_f16 v92, v77, v81
.Ltmp11248:
	v_dot2c_f32_f16 v18, v78, v82
.Ltmp11249:
	v_dot2c_f32_f16 v92, v79, v83
.Ltmp11250:
	ds_read_b128 v[76:79], v44 offset:256
.Ltmp11251:
	ds_read_b128 v[80:83], v43 offset:8448
.Ltmp11252:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v84, v88
.Ltmp11253:
	v_dot2c_f32_f16 v92, v85, v89
.Ltmp11254:
	v_dot2c_f32_f16 v18, v86, v90
.Ltmp11255:
	v_dot2c_f32_f16 v92, v87, v91
.Ltmp11256:
	ds_read_b128 v[84:87], v44 offset:272
.Ltmp11257:
	ds_read_b128 v[88:91], v43 offset:8464
.Ltmp11258:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v29, v72
.Ltmp11259:
	v_dot2c_f32_f16 v92, v30, v73
.Ltmp11260:
	v_dot2c_f32_f16 v18, v31, v74
.Ltmp11261:
	v_dot2c_f32_f16 v92, v32, v75
.Ltmp11262:
	ds_read_b128 v[29:32], v44 offset:288
.Ltmp11263:
	ds_read_b128 v[72:75], v43 offset:8480
.Ltmp11264:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v76, v80
.Ltmp11265:
	v_dot2c_f32_f16 v92, v77, v81
.Ltmp11266:
	v_dot2c_f32_f16 v18, v78, v82
.Ltmp11267:
	v_dot2c_f32_f16 v92, v79, v83
.Ltmp11268:
	ds_read_b128 v[76:79], v44 offset:304
.Ltmp11269:
	ds_read_b128 v[80:83], v43 offset:8496
.Ltmp11270:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v84, v88
.Ltmp11271:
	v_dot2c_f32_f16 v92, v85, v89
.Ltmp11272:
	v_dot2c_f32_f16 v18, v86, v90
.Ltmp11273:
	v_dot2c_f32_f16 v92, v87, v91
.Ltmp11274:
	ds_read_b128 v[84:87], v44 offset:320
.Ltmp11275:
	ds_read_b128 v[88:91], v43 offset:8512
.Ltmp11276:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v29, v72
.Ltmp11277:
	v_dot2c_f32_f16 v92, v30, v73
.Ltmp11278:
	v_dot2c_f32_f16 v18, v31, v74
.Ltmp11279:
	v_dot2c_f32_f16 v92, v32, v75
.Ltmp11280:
	ds_read_b128 v[29:32], v44 offset:336
.Ltmp11281:
	ds_read_b128 v[72:75], v43 offset:8528
.Ltmp11282:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v76, v80
.Ltmp11283:
	v_dot2c_f32_f16 v92, v77, v81
.Ltmp11284:
	v_dot2c_f32_f16 v18, v78, v82
.Ltmp11285:
	v_dot2c_f32_f16 v92, v79, v83
.Ltmp11286:
	ds_read_b128 v[76:79], v44 offset:352
.Ltmp11287:
	ds_read_b128 v[80:83], v43 offset:8544
.Ltmp11288:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v84, v88
.Ltmp11289:
	v_dot2c_f32_f16 v92, v85, v89
.Ltmp11290:
	v_dot2c_f32_f16 v18, v86, v90
.Ltmp11291:
	v_dot2c_f32_f16 v92, v87, v91
.Ltmp11292:
	ds_read_b128 v[84:87], v44 offset:368
.Ltmp11293:
	ds_read_b128 v[88:91], v43 offset:8560
.Ltmp11294:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v29, v72
.Ltmp11295:
	v_dot2c_f32_f16 v92, v30, v73
.Ltmp11296:
	v_dot2c_f32_f16 v18, v31, v74
.Ltmp11297:
	v_dot2c_f32_f16 v92, v32, v75
.Ltmp11298:
	ds_read_b128 v[29:32], v44 offset:384
.Ltmp11299:
	ds_read_b128 v[72:75], v43 offset:8576
.Ltmp11300:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v76, v80
.Ltmp11301:
	v_dot2c_f32_f16 v92, v77, v81
.Ltmp11302:
	v_dot2c_f32_f16 v18, v78, v82
.Ltmp11303:
	v_dot2c_f32_f16 v92, v79, v83
.Ltmp11304:
	ds_read_b128 v[76:79], v44 offset:400
.Ltmp11305:
	ds_read_b128 v[80:83], v43 offset:8592
.Ltmp11306:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v84, v88
.Ltmp11307:
	v_dot2c_f32_f16 v92, v85, v89
.Ltmp11308:
	v_dot2c_f32_f16 v18, v86, v90
.Ltmp11309:
	v_dot2c_f32_f16 v92, v87, v91
.Ltmp11310:
	ds_read_b128 v[84:87], v44 offset:416
.Ltmp11311:
	ds_read_b128 v[88:91], v43 offset:8608
.Ltmp11312:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v29, v72
.Ltmp11313:
	v_dot2c_f32_f16 v92, v30, v73
.Ltmp11314:
	v_dot2c_f32_f16 v18, v31, v74
.Ltmp11315:
	v_dot2c_f32_f16 v92, v32, v75
.Ltmp11316:
	ds_read_b128 v[29:32], v44 offset:432
.Ltmp11317:
	ds_read_b128 v[72:75], v43 offset:8624
.Ltmp11318:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v76, v80
.Ltmp11319:
	v_dot2c_f32_f16 v92, v77, v81
.Ltmp11320:
	v_dot2c_f32_f16 v18, v78, v82
.Ltmp11321:
	v_dot2c_f32_f16 v92, v79, v83
.Ltmp11322:
	ds_read_b128 v[76:79], v44 offset:448
.Ltmp11323:
	ds_read_b128 v[80:83], v43 offset:8640
.Ltmp11324:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v84, v88
.Ltmp11325:
	v_dot2c_f32_f16 v92, v85, v89
.Ltmp11326:
	v_dot2c_f32_f16 v18, v86, v90
.Ltmp11327:
	v_dot2c_f32_f16 v92, v87, v91
.Ltmp11328:
	ds_read_b128 v[84:87], v44 offset:464
.Ltmp11329:
	ds_read_b128 v[88:91], v43 offset:8656
.Ltmp11330:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v29, v72
.Ltmp11331:
	v_dot2c_f32_f16 v92, v30, v73
.Ltmp11332:
	v_dot2c_f32_f16 v18, v31, v74
.Ltmp11333:
	v_dot2c_f32_f16 v92, v32, v75
.Ltmp11334:
	ds_read_b128 v[29:32], v44 offset:480
.Ltmp11335:
	ds_read_b128 v[72:75], v43 offset:8672
.Ltmp11336:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v76, v80
.Ltmp11337:
	v_dot2c_f32_f16 v92, v77, v81
.Ltmp11338:
	v_dot2c_f32_f16 v18, v78, v82
.Ltmp11339:
	v_dot2c_f32_f16 v92, v79, v83
.Ltmp11340:
	ds_read_b128 v[76:79], v44 offset:496
.Ltmp11341:
	ds_read_b128 v[80:83], v43 offset:8688
.Ltmp11342:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v84, v88
.Ltmp11343:
	v_dot2c_f32_f16 v92, v85, v89
.Ltmp11344:
	v_dot2c_f32_f16 v18, v86, v90
.Ltmp11345:
	v_dot2c_f32_f16 v92, v87, v91
.Ltmp11346:
	s_waitcnt lgkmcnt(2)
	v_dot2c_f32_f16 v18, v29, v72
.Ltmp11347:
	v_dot2c_f32_f16 v92, v30, v73
.Ltmp11348:
	v_dot2c_f32_f16 v18, v31, v74
.Ltmp11349:
	v_dot2c_f32_f16 v92, v32, v75
.Ltmp11350:
	s_waitcnt lgkmcnt(0)
	v_dot2c_f32_f16 v18, v76, v80
.Ltmp11351:
	v_dot2c_f32_f16 v92, v77, v81
.Ltmp11352:
	v_dot2c_f32_f16 v18, v78, v82
.Ltmp11353:
	v_dot2c_f32_f16 v92, v79, v83
.Ltmp11354:
	v_add_f32_e32 v18, v92, v18
.Ltmp11355:
	v_mul_f32_e32 v18, s29, v18
.Ltmp11356:
.LBB19_64:
	s_or_b32 exec_lo, exec_lo, s3
.Ltmp11357:
.LBB19_65:
	s_or_b32 exec_lo, exec_lo, s4
.Ltmp11358:
	ds_bpermute_b32 v29, v60, v18
.Ltmp11359:
	v_max_f32_e32 v30, v18, v18
	v_mov_b32_e32 v31, 0
.Ltmp11360:
	s_mov_b32 s3, exec_lo
.Ltmp11361:
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v29, v29, v29
.Ltmp11362:
	v_max_f32_e32 v29, v30, v29
.Ltmp11363:
	ds_bpermute_b32 v30, v61, v29
.Ltmp11364:
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v30, v30, v30
.Ltmp11365:
	v_max_f32_e32 v29, v29, v30
.Ltmp11366:
	ds_bpermute_b32 v30, v62, v29
.Ltmp11367:
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v30, v30, v30
.Ltmp11368:
	v_max_f32_e32 v29, v29, v30
.Ltmp11369:
	ds_bpermute_b32 v30, v63, v29
.Ltmp11370:
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v30, v30, v30
.Ltmp11371:
	v_max_f32_e32 v30, v29, v30
.Ltmp11372:
	v_mov_b32_e32 v29, 1.0
.Ltmp11373:
	v_cmpx_lg_f32_e32 0xff800000, v30
	s_cbranch_execz .LBB19_69
.Ltmp11374:
	v_max_f32_e32 v29, v30, v30
	v_max_f32_e32 v30, v42, v42
.Ltmp11375:
	s_mov_b32 s4, exec_lo
.Ltmp11376:
	v_max_f32_e32 v30, v30, v29
.Ltmp11377:
	v_mov_b32_e32 v29, 0
.Ltmp11378:
	v_cmpx_neq_f32_e32 0xff800000, v42
	s_cbranch_execz .LBB19_68
.Ltmp11379:
	v_sub_f32_e32 v29, v42, v30
.Ltmp11380:
	v_mul_f32_e32 v31, 0x3fb8aa3b, v29
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v29
	v_fma_f32 v32, 0x3fb8aa3b, v29, -v31
	v_rndne_f32_e32 v42, v31
.Ltmp11381:
	v_fmac_f32_e32 v32, 0x32a5705f, v29
	v_sub_f32_e32 v31, v31, v42
	v_add_f32_e32 v31, v31, v32
	v_cvt_i32_f32_e32 v32, v42
	v_exp_f32_e32 v31, v31
	v_ldexp_f32 v31, v31, v32
	v_cndmask_b32_e32 v31, 0, v31, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v29
	v_cndmask_b32_e32 v29, 0x7f800000, v31, vcc_lo
.Ltmp11382:
.LBB19_68:
	s_or_b32 exec_lo, exec_lo, s4
.Ltmp11383:
	v_sub_f32_e32 v18, v18, v30
.Ltmp11384:
	v_mov_b32_e32 v42, v30
.Ltmp11385:
	v_mul_f32_e32 v18, 0x3fb8aa3b, v18
.Ltmp11386:
	v_exp_f32_e32 v31, v18
.Ltmp11387:
.LBB19_69:
	s_or_b32 exec_lo, exec_lo, s3
.Ltmp11388:
	ds_bpermute_b32 v18, v60, v31
.Ltmp11389:
	v_mul_f32_e32 v2, v29, v2
.Ltmp11390:
	v_mul_f32_e32 v3, v29, v3
.Ltmp11391:
	v_mul_f32_e32 v4, v29, v4
.Ltmp11392:
	v_mul_f32_e32 v5, v29, v5
.Ltmp11393:
	v_mul_f32_e32 v6, v29, v6
.Ltmp11394:
	v_mul_f32_e32 v7, v29, v7
.Ltmp11395:
	v_mul_f32_e32 v8, v29, v8
.Ltmp11396:
	v_mul_f32_e32 v9, v29, v9
.Ltmp11397:
	v_mul_f32_e32 v10, v29, v10
.Ltmp11398:
	v_mul_f32_e32 v11, v29, v11
.Ltmp11399:
	v_mul_f32_e32 v12, v29, v12
.Ltmp11400:
	v_mul_f32_e32 v13, v29, v13
.Ltmp11401:
	v_mul_f32_e32 v14, v29, v14
.Ltmp11402:
	v_mul_f32_e32 v15, v29, v15
.Ltmp11403:
	v_mul_f32_e32 v16, v29, v16
.Ltmp11404:
	v_mul_f32_e32 v17, v29, v17
.Ltmp11405:
	ds_write_b32 v46, v31 offset:25088
.Ltmp11406:
	s_waitcnt lgkmcnt(0)
	s_barrier
.Ltmp11407:
	v_add_f32_e32 v18, v31, v18
.Ltmp11408:
	buffer_gl0_inv
.Ltmp11409:
	ds_bpermute_b32 v30, v61, v18
.Ltmp11410:
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v18, v18, v30
.Ltmp11411:
	ds_bpermute_b32 v30, v62, v18
.Ltmp11412:
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v18, v18, v30
.Ltmp11413:
	ds_bpermute_b32 v30, v63, v18
.Ltmp11414:
	s_and_saveexec_b32 s3, s1
	s_cbranch_execz .LBB19_14
.Ltmp11415:
	s_cmp_lt_i32 s37, 1
	s_cbranch_scc1 .LBB19_72
.Ltmp11416:
	ds_read_b32 v31, v47 offset:25088
.Ltmp11417:
	ds_read_b128 v[72:75], v38 offset:16640
.Ltmp11418:
	ds_read_b128 v[76:79], v38 offset:16656
.Ltmp11419:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp11420:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11421:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp11422:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11423:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp11424:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11425:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp11426:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11427:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp11428:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11429:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp11430:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11431:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp11432:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11433:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp11434:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11435:
.LBB19_72:
	s_cmp_lt_i32 s37, 2
	s_cbranch_scc0 .LBB19_87
.Ltmp11436:
	s_cmp_lt_i32 s37, 3
	s_cbranch_scc0 .LBB19_88
.Ltmp11437:
.LBB19_74:
	s_cmp_lt_i32 s37, 4
	s_cbranch_scc0 .LBB19_89
.Ltmp11438:
.LBB19_75:
	s_cmp_lt_i32 s37, 5
	s_cbranch_scc0 .LBB19_90
.Ltmp11439:
.LBB19_76:
	s_cmp_lt_i32 s37, 6
	s_cbranch_scc0 .LBB19_91
.Ltmp11440:
.LBB19_77:
	s_cmp_lt_i32 s37, 7
	s_cbranch_scc0 .LBB19_92
.Ltmp11441:
.LBB19_78:
	s_cmp_lt_i32 s37, 8
	s_cbranch_scc0 .LBB19_93
.Ltmp11442:
.LBB19_79:
	s_cmp_lt_i32 s37, 9
	s_cbranch_scc0 .LBB19_94
.Ltmp11443:
.LBB19_80:
	s_cmp_lt_i32 s37, 10
	s_cbranch_scc0 .LBB19_95
.Ltmp11444:
.LBB19_81:
	s_cmp_lt_i32 s37, 11
	s_cbranch_scc0 .LBB19_96
.Ltmp11445:
.LBB19_82:
	s_cmp_lt_i32 s37, 12
	s_cbranch_scc0 .LBB19_97
.Ltmp11446:
.LBB19_83:
	s_cmp_lt_i32 s37, 13
	s_cbranch_scc0 .LBB19_98
.Ltmp11447:
.LBB19_84:
	s_cmp_lt_i32 s37, 14
	s_cbranch_scc0 .LBB19_99
.Ltmp11448:
.LBB19_85:
	s_cmp_lt_i32 s37, 15
	s_cbranch_scc0 .LBB19_100
.Ltmp11449:
.LBB19_86:
	s_cmp_lt_i32 s37, 16
	s_cbranch_scc1 .LBB19_14
	s_branch .LBB19_101
.Ltmp11450:
.LBB19_87:
	ds_read_b32 v31, v47 offset:25092
.Ltmp11451:
	ds_read_b128 v[72:75], v38 offset:17168
.Ltmp11452:
	ds_read_b128 v[76:79], v38 offset:17184
.Ltmp11453:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp11454:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11455:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp11456:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11457:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp11458:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11459:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp11460:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11461:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp11462:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11463:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp11464:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11465:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp11466:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11467:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp11468:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11469:
	s_cmp_lt_i32 s37, 3
	s_cbranch_scc1 .LBB19_74
.Ltmp11470:
.LBB19_88:
	ds_read_b32 v31, v47 offset:25096
.Ltmp11471:
	ds_read_b128 v[72:75], v38 offset:17696
.Ltmp11472:
	ds_read_b128 v[76:79], v38 offset:17712
.Ltmp11473:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp11474:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11475:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp11476:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11477:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp11478:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11479:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp11480:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11481:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp11482:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11483:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp11484:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11485:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp11486:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11487:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp11488:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11489:
	s_cmp_lt_i32 s37, 4
	s_cbranch_scc1 .LBB19_75
.Ltmp11490:
.LBB19_89:
	ds_read_b32 v31, v47 offset:25100
.Ltmp11491:
	ds_read_b128 v[72:75], v38 offset:18224
.Ltmp11492:
	ds_read_b128 v[76:79], v38 offset:18240
.Ltmp11493:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp11494:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11495:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp11496:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11497:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp11498:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11499:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp11500:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11501:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp11502:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11503:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp11504:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11505:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp11506:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11507:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp11508:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11509:
	s_cmp_lt_i32 s37, 5
	s_cbranch_scc1 .LBB19_76
.Ltmp11510:
.LBB19_90:
	ds_read_b32 v31, v47 offset:25104
.Ltmp11511:
	ds_read_b128 v[72:75], v38 offset:18752
.Ltmp11512:
	ds_read_b128 v[76:79], v38 offset:18768
.Ltmp11513:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp11514:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11515:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp11516:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11517:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp11518:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11519:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp11520:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11521:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp11522:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11523:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp11524:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11525:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp11526:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11527:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp11528:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11529:
	s_cmp_lt_i32 s37, 6
	s_cbranch_scc1 .LBB19_77
.Ltmp11530:
.LBB19_91:
	ds_read_b32 v31, v47 offset:25108
.Ltmp11531:
	ds_read_b128 v[72:75], v38 offset:19280
.Ltmp11532:
	ds_read_b128 v[76:79], v38 offset:19296
.Ltmp11533:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp11534:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11535:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp11536:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11537:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp11538:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11539:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp11540:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11541:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp11542:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11543:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp11544:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11545:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp11546:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11547:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp11548:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11549:
	s_cmp_lt_i32 s37, 7
	s_cbranch_scc1 .LBB19_78
.Ltmp11550:
.LBB19_92:
	ds_read_b32 v31, v47 offset:25112
.Ltmp11551:
	ds_read_b128 v[72:75], v38 offset:19808
.Ltmp11552:
	ds_read_b128 v[76:79], v38 offset:19824
.Ltmp11553:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp11554:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11555:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp11556:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11557:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp11558:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11559:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp11560:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11561:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp11562:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11563:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp11564:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11565:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp11566:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11567:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp11568:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11569:
	s_cmp_lt_i32 s37, 8
	s_cbranch_scc1 .LBB19_79
.Ltmp11570:
.LBB19_93:
	ds_read_b32 v31, v47 offset:25116
.Ltmp11571:
	ds_read_b128 v[72:75], v38 offset:20336
.Ltmp11572:
	ds_read_b128 v[76:79], v38 offset:20352
.Ltmp11573:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp11574:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11575:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp11576:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11577:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp11578:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11579:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp11580:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11581:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp11582:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11583:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp11584:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11585:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp11586:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11587:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp11588:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11589:
	s_cmp_lt_i32 s37, 9
	s_cbranch_scc1 .LBB19_80
.Ltmp11590:
.LBB19_94:
	ds_read_b32 v31, v47 offset:25120
.Ltmp11591:
	ds_read_b128 v[72:75], v38 offset:20864
.Ltmp11592:
	ds_read_b128 v[76:79], v38 offset:20880
.Ltmp11593:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp11594:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11595:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp11596:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11597:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp11598:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11599:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp11600:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11601:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp11602:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11603:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp11604:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11605:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp11606:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11607:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp11608:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11609:
	s_cmp_lt_i32 s37, 10
	s_cbranch_scc1 .LBB19_81
.Ltmp11610:
.LBB19_95:
	ds_read_b32 v31, v47 offset:25124
.Ltmp11611:
	ds_read_b128 v[72:75], v38 offset:21392
.Ltmp11612:
	ds_read_b128 v[76:79], v38 offset:21408
.Ltmp11613:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp11614:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11615:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp11616:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11617:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp11618:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11619:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp11620:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11621:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp11622:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11623:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp11624:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11625:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp11626:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11627:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp11628:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11629:
	s_cmp_lt_i32 s37, 11
	s_cbranch_scc1 .LBB19_82
.Ltmp11630:
.LBB19_96:
	ds_read_b32 v31, v47 offset:25128
.Ltmp11631:
	ds_read_b128 v[72:75], v38 offset:21920
.Ltmp11632:
	ds_read_b128 v[76:79], v38 offset:21936
.Ltmp11633:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp11634:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11635:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp11636:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11637:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp11638:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11639:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp11640:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11641:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp11642:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11643:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp11644:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11645:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp11646:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11647:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp11648:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11649:
	s_cmp_lt_i32 s37, 12
	s_cbranch_scc1 .LBB19_83
.Ltmp11650:
.LBB19_97:
	ds_read_b32 v31, v47 offset:25132
.Ltmp11651:
	ds_read_b128 v[72:75], v38 offset:22448
.Ltmp11652:
	ds_read_b128 v[76:79], v38 offset:22464
.Ltmp11653:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp11654:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11655:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp11656:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11657:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp11658:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11659:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp11660:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11661:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp11662:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11663:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp11664:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11665:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp11666:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11667:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp11668:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11669:
	s_cmp_lt_i32 s37, 13
	s_cbranch_scc1 .LBB19_84
.Ltmp11670:
.LBB19_98:
	ds_read_b32 v31, v47 offset:25136
.Ltmp11671:
	ds_read_b128 v[72:75], v38 offset:22976
.Ltmp11672:
	ds_read_b128 v[76:79], v38 offset:22992
.Ltmp11673:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp11674:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11675:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp11676:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11677:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp11678:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11679:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp11680:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11681:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp11682:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11683:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp11684:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11685:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp11686:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11687:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp11688:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11689:
	s_cmp_lt_i32 s37, 14
	s_cbranch_scc1 .LBB19_85
.Ltmp11690:
.LBB19_99:
	ds_read_b32 v31, v47 offset:25140
.Ltmp11691:
	ds_read_b128 v[72:75], v38 offset:23504
.Ltmp11692:
	ds_read_b128 v[76:79], v38 offset:23520
.Ltmp11693:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp11694:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11695:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp11696:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11697:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp11698:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11699:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp11700:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11701:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp11702:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11703:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp11704:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11705:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp11706:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11707:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp11708:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11709:
	s_cmp_lt_i32 s37, 15
	s_cbranch_scc1 .LBB19_86
.Ltmp11710:
.LBB19_100:
	ds_read_b32 v31, v47 offset:25144
.Ltmp11711:
	ds_read_b128 v[72:75], v38 offset:24032
.Ltmp11712:
	ds_read_b128 v[76:79], v38 offset:24048
.Ltmp11713:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp11714:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11715:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp11716:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11717:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp11718:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11719:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp11720:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11721:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp11722:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11723:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp11724:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11725:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp11726:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11727:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp11728:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11729:
	s_cmp_lt_i32 s37, 16
	s_cbranch_scc1 .LBB19_14
.Ltmp11730:
.LBB19_101:
	ds_read_b32 v31, v47 offset:25148
.Ltmp11731:
	ds_read_b128 v[72:75], v38 offset:24560
.Ltmp11732:
	ds_read_b128 v[76:79], v38 offset:24576
.Ltmp11733:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp11734:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11735:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp11736:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11737:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp11738:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11739:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp11740:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11741:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp11742:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11743:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp11744:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11745:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp11746:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11747:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp11748:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp11749:
	s_branch .LBB19_14
.Ltmp11750:
.LBB19_102:
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, v18
	v_mov_b32_e32 v20, v18
	v_mov_b32_e32 v21, v18
	v_mov_b32_e32 v22, v18
	v_mov_b32_e32 v23, v18
	v_mov_b32_e32 v24, v18
	v_mov_b32_e32 v25, v18
	v_mov_b32_e32 v26, v18
	v_mov_b32_e32 v27, v18
	v_mov_b32_e32 v28, v18
	v_mov_b32_e32 v29, v18
	v_mov_b32_e32 v30, v18
	v_mov_b32_e32 v31, v18
	v_mov_b32_e32 v32, v18
	v_mov_b32_e32 v33, v18
.Ltmp11751:
	v_mov_b32_e32 v2, v18
	v_mov_b32_e32 v3, v19
	v_mov_b32_e32 v4, v20
	v_mov_b32_e32 v5, v21
	v_mov_b32_e32 v6, v22
	v_mov_b32_e32 v7, v23
	v_mov_b32_e32 v8, v24
	v_mov_b32_e32 v9, v25
	v_mov_b32_e32 v10, v26
	v_mov_b32_e32 v11, v27
	v_mov_b32_e32 v12, v28
	v_mov_b32_e32 v13, v29
	v_mov_b32_e32 v14, v30
	v_mov_b32_e32 v15, v31
	v_mov_b32_e32 v16, v32
	v_mov_b32_e32 v17, v33
.Ltmp11752:
.LBB19_103:
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s42, v35
	s_cbranch_execz .LBB19_105
.Ltmp11753:
	v_div_scale_f32 v1, null, v18, v18, 1.0
	v_or_b32_e32 v19, s41, v35
	v_div_scale_f32 v22, vcc_lo, 1.0, v18, 1.0
	v_rcp_f32_e32 v21, v1
.Ltmp11754:
	v_add_nc_u32_e32 v19, s40, v19
	v_mul_lo_u32 v19, v19, s11
	v_fma_f32 v20, -v1, v21, 1.0
.Ltmp11755:
	v_fmac_f32_e32 v21, v20, v21
	v_lshrrev_b32_e32 v20, 7, v0
.Ltmp11756:
	v_mul_f32_e32 v23, v22, v21
	v_add_nc_u32_e32 v20, s28, v20
.Ltmp11757:
	v_fma_f32 v24, -v1, v23, v22
	v_add_lshl_u32 v19, v20, v19, 8
	v_fmac_f32_e32 v23, v24, v21
	v_ashrrev_i32_e32 v20, 31, v19
	v_fma_f32 v1, -v1, v23, v22
	v_lshlrev_b64 v[19:20], 1, v[19:20]
	v_lshlrev_b32_e32 v22, 1, v34
	v_div_fmas_f32 v1, v1, v21, v23
	v_add_co_u32 v19, vcc_lo, s34, v19
	v_add_co_ci_u32_e64 v20, null, s35, v20, vcc_lo
	v_div_fixup_f32 v1, v1, v18, 1.0
.Ltmp11758:
	v_add_co_u32 v26, vcc_lo, v19, v22
	v_add_co_ci_u32_e64 v27, null, 0, v20, vcc_lo
.Ltmp11759:
	v_fma_mixlo_f16 v18, v1, v2, 0
.Ltmp11760:
	v_fma_mixlo_f16 v19, v1, v4, 0
.Ltmp11761:
	v_fma_mixlo_f16 v20, v1, v6, 0
.Ltmp11762:
	v_fma_mixlo_f16 v21, v1, v8, 0
	v_fma_mixlo_f16 v22, v1, v10, 0
.Ltmp11763:
	v_fma_mixlo_f16 v23, v1, v12, 0
.Ltmp11764:
	v_fma_mixlo_f16 v24, v1, v14, 0
.Ltmp11765:
	v_fma_mixlo_f16 v25, v1, v16, 0
.Ltmp11766:
	v_fma_mixhi_f16 v21, v1, v9, 0
	v_fma_mixhi_f16 v20, v1, v7, 0
	v_fma_mixhi_f16 v19, v1, v5, 0
	v_fma_mixhi_f16 v18, v1, v3, 0
	v_fma_mixhi_f16 v25, v1, v17, 0
	v_fma_mixhi_f16 v24, v1, v15, 0
	v_fma_mixhi_f16 v23, v1, v13, 0
	v_fma_mixhi_f16 v22, v1, v11, 0
	global_store_dwordx4 v[26:27], v[18:21], off
.Ltmp11767:
	global_store_dwordx4 v[26:27], v[22:25], off offset:16
.Ltmp11768:
.LBB19_105:
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.Ltmp11769:
.LBB19_106:
	s_and_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccz .LBB19_115
.Ltmp11770:
	s_lshl_b32 s3, s33, 9
.Ltmp11771:
	s_mov_b32 s0, exec_lo
.Ltmp11772:
	v_cmpx_gt_i32_e64 s3, v0
.Ltmp11773:
	s_cbranch_execz .LBB19_115
.Ltmp11774:
	v_xad_u32 v1, v0, -1, s3
	s_lshl_b32 s5, s33, 8
	s_add_i32 s4, s40, s41
	s_mov_b32 s0, -1
	s_mov_b32 s7, 0
	v_lshrrev_b32_e32 v2, 8, v1
	v_mov_b32_e32 v1, v0
	s_mov_b32 s6, exec_lo
	v_cmpx_ne_u32_e32 0, v2
	s_cbranch_execz .LBB19_112
.Ltmp11775:
	s_abs_i32 s8, s5
	s_abs_i32 s9, s33
	v_cvt_f32_u32_e32 v1, s8
	v_cvt_f32_u32_e32 v3, s9
	s_sub_i32 s0, 0, s8
	s_sub_i32 s1, 0, s9
	v_mov_b32_e32 v5, v0
	v_rcp_iflag_f32_e32 v1, v1
	v_rcp_iflag_f32_e32 v3, v3
	v_mov_b32_e32 v6, 0
	s_mov_b32 s10, s4
	s_mov_b32 s12, s11
	s_mov_b32 s13, s28
	s_ashr_i32 s14, s5, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_mul_f32_e32 v3, 0x4f7ffffe, v3
	v_cvt_u32_f32_e32 v7, v1
	v_cvt_u32_f32_e32 v8, v3
	v_add_nc_u32_e32 v1, 1, v2
	v_mul_lo_u32 v3, s0, v7
	v_mul_lo_u32 v4, s1, v8
	v_and_b32_e32 v2, 0x1fffffe, v1
	v_mul_hi_u32 v9, v7, v3
	v_mul_hi_u32 v10, v8, v4
	v_mov_b32_e32 v3, v0
	v_or_b32_e32 v4, 0x100, v0
	v_add_nc_u32_e32 v7, v7, v9
	v_add_nc_u32_e32 v8, v8, v10
	v_mov_b32_e32 v9, v2
.Ltmp11776:
.LBB19_110:
	v_mul_hi_u32 v10, v5, v7
	v_mul_hi_u32 v13, v4, v7
	v_lshrrev_b32_e32 v11, 8, v5
	v_lshrrev_b32_e32 v12, 8, v4
.Ltmp11777:
	v_add_nc_u32_e32 v9, -2, v9
.Ltmp11778:
	v_mul_hi_u32 v14, v11, v8
	v_mul_lo_u32 v16, v10, s8
	v_mul_lo_u32 v17, v13, s8
	v_mul_hi_u32 v15, v12, v8
	v_add_nc_u32_e32 v18, 1, v10
	v_add_nc_u32_e32 v19, 1, v13
.Ltmp11779:
	v_cmp_eq_u32_e32 vcc_lo, 0, v9
.Ltmp11780:
	v_mul_lo_u32 v14, v14, s9
	v_sub_nc_u32_e32 v16, v5, v16
	v_sub_nc_u32_e32 v17, v4, v17
	v_mul_lo_u32 v15, v15, s9
	v_add_nc_u32_e32 v4, 0x200, v4
	v_add_nc_u32_e32 v5, 0x200, v5
	v_cmp_le_u32_e64 s0, s8, v16
	v_cmp_le_u32_e64 s1, s8, v17
	v_sub_nc_u32_e32 v11, v11, v14
.Ltmp11781:
	s_or_b32 s7, vcc_lo, s7
.Ltmp11782:
	v_sub_nc_u32_e32 v12, v12, v15
	v_cndmask_b32_e64 v10, v10, v18, s0
	v_subrev_nc_u32_e32 v18, s8, v16
	v_cndmask_b32_e64 v13, v13, v19, s1
	v_subrev_nc_u32_e32 v19, s8, v17
	v_add_nc_u32_e32 v15, 1, v10
	v_cndmask_b32_e64 v14, v16, v18, s0
	v_subrev_nc_u32_e32 v18, s9, v11
	v_cmp_le_u32_e64 s0, s9, v11
	v_cndmask_b32_e64 v16, v17, v19, s1
	v_subrev_nc_u32_e32 v19, s9, v12
	v_cmp_le_u32_e64 s1, s9, v12
	v_add_nc_u32_e32 v17, 1, v13
	v_cndmask_b32_e64 v11, v11, v18, s0
	v_cmp_le_u32_e64 s2, s8, v14
	v_cmp_le_u32_e64 s0, s8, v16
	v_cndmask_b32_e64 v12, v12, v19, s1
	v_subrev_nc_u32_e32 v14, s9, v11
	v_cmp_le_u32_e64 s1, s9, v11
	v_cndmask_b32_e64 v10, v10, v15, s2
	v_cndmask_b32_e64 v13, v13, v17, s0
	v_subrev_nc_u32_e32 v15, s9, v12
	v_cmp_le_u32_e64 s0, s9, v12
	v_cndmask_b32_e64 v11, v11, v14, s1
	v_xor_b32_e32 v10, s14, v10
	v_xor_b32_e32 v13, s14, v13
	v_cndmask_b32_e64 v12, v12, v15, s0
	v_add_nc_u32_e32 v11, s4, v11
	v_subrev_nc_u32_e32 v10, s14, v10
	v_subrev_nc_u32_e32 v13, s14, v13
	v_add_nc_u32_e32 v12, s10, v12
	v_mul_lo_u32 v11, v11, s11
	v_add_nc_u32_e32 v10, s28, v10
	v_add_nc_u32_e32 v13, s13, v13
	v_mul_lo_u32 v12, v12, s12
	v_add_lshl_u32 v10, v10, v11, 8
	v_add_lshl_u32 v11, v13, v12, 8
	v_or_b32_e32 v10, v10, v0
	v_or_b32_e32 v12, v11, v3
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v13, 31, v12
	v_lshlrev_b64 v[10:11], 1, v[10:11]
	v_lshlrev_b64 v[12:13], 1, v[12:13]
	v_add_co_u32 v10, s0, s34, v10
	v_add_co_ci_u32_e64 v11, null, s35, v11, s0
	v_add_co_u32 v12, s0, s34, v12
	v_add_co_ci_u32_e64 v13, null, s35, v13, s0
	global_store_short v[10:11], v6, off
	global_store_short v[12:13], v6, off
.Ltmp11783:
	s_andn2_b32 exec_lo, exec_lo, s7
	s_cbranch_execnz .LBB19_110
.Ltmp11784:
	s_or_b32 exec_lo, exec_lo, s7
	v_cmp_ne_u32_e32 vcc_lo, v1, v2
	v_lshl_or_b32 v1, v2, 8, v0
	s_orn2_b32 s0, vcc_lo, exec_lo
.Ltmp11785:
.LBB19_112:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 exec_lo, exec_lo, s0
	s_cbranch_execz .LBB19_115
.Ltmp11786:
	s_abs_i32 s1, s5
	v_cvt_f32_u32_e32 v2, s33
	v_cvt_f32_u32_e32 v3, s1
	s_sub_i32 s0, 0, s33
	s_sub_i32 s2, 0, s1
	v_rcp_iflag_f32_e32 v2, v2
	v_rcp_iflag_f32_e32 v3, v3
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_mul_f32_e32 v3, 0x4f7ffffe, v3
	v_cvt_u32_f32_e32 v4, v2
	v_cvt_u32_f32_e32 v5, v3
	v_mul_lo_u32 v2, s0, v4
	v_mul_lo_u32 v3, s2, v5
	s_ashr_i32 s2, s5, 31
	s_mov_b32 s5, 0
	v_mul_hi_u32 v6, v4, v2
	v_mul_hi_u32 v7, v5, v3
	v_mov_b32_e32 v2, 0
	v_add_nc_u32_e32 v3, v4, v6
	v_add_nc_u32_e32 v4, v5, v7
	v_lshrrev_b32_e32 v5, 8, v1
.Ltmp11787:
.LBB19_114:
	v_sub_nc_u32_e32 v6, 0, v1
.Ltmp11788:
	v_mul_hi_u32 v7, v5, v3
	v_ashrrev_i32_e32 v9, 31, v1
	v_max_i32_e32 v6, v6, v1
.Ltmp11789:
	v_add_nc_u32_e32 v1, 0x100, v1
.Ltmp11790:
	v_xor_b32_e32 v9, s2, v9
	v_mul_lo_u32 v7, v7, s33
	v_mul_hi_u32 v8, v6, v4
	v_sub_nc_u32_e32 v7, v5, v7
	v_mul_lo_u32 v10, v8, s1
	v_add_nc_u32_e32 v11, 1, v8
.Ltmp11791:
	v_add_nc_u32_e32 v5, 1, v5
.Ltmp11792:
	v_subrev_nc_u32_e32 v12, s33, v7
	v_cmp_le_u32_e32 vcc_lo, s33, v7
	v_sub_nc_u32_e32 v6, v6, v10
	v_cndmask_b32_e32 v7, v7, v12, vcc_lo
	v_subrev_nc_u32_e32 v10, s1, v6
	v_cmp_le_u32_e32 vcc_lo, s1, v6
	v_cmp_le_u32_e64 s0, s33, v7
	v_cndmask_b32_e32 v8, v8, v11, vcc_lo
	v_cndmask_b32_e32 v6, v6, v10, vcc_lo
	v_subrev_nc_u32_e32 v11, s33, v7
	v_add_nc_u32_e32 v10, 1, v8
	v_cmp_le_u32_e32 vcc_lo, s1, v6
	v_cndmask_b32_e64 v7, v7, v11, s0
.Ltmp11793:
	v_cndmask_b32_e32 v6, v8, v10, vcc_lo
	v_add_nc_u32_e32 v7, s4, v7
.Ltmp11794:
	v_cmp_le_i32_e32 vcc_lo, s3, v1
.Ltmp11795:
	v_xor_b32_e32 v6, v6, v9
	v_mul_lo_u32 v7, v7, s11
.Ltmp11796:
	s_or_b32 s5, vcc_lo, s5
.Ltmp11797:
	v_sub_nc_u32_e32 v6, v6, v9
.Ltmp11798:
	v_add3_u32 v6, v6, s28, v7
.Ltmp11799:
	v_lshl_or_b32 v6, v6, 8, v0
	v_ashrrev_i32_e32 v7, 31, v6
.Ltmp11800:
	v_lshlrev_b64 v[6:7], 1, v[6:7]
	v_add_co_u32 v6, s0, s34, v6
	v_add_co_ci_u32_e64 v7, null, s35, v7, s0
	global_store_short v[6:7], v2, off
.Ltmp11801:
	s_andn2_b32 exec_lo, exec_lo, s5
	s_cbranch_execnz .LBB19_114
.Ltmp11802:
.LBB19_115:
	s_endpgm
.Ltmp11803:
.Lfunc_end19:
