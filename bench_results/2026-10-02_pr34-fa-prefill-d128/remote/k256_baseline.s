SYM:
.Lfunc_begin17:
	s_load_dwordx4 s[0:3], s[4:5], 0x20
.Ltmp9909:
	s_ashr_i32 s9, s8, 31
	s_lshl_b64 s[10:11], s[8:9], 2
	s_waitcnt lgkmcnt(0)
	s_add_u32 s0, s0, s10
.Ltmp9910:
	s_addc_u32 s1, s1, s11
	s_load_dwordx2 s[40:41], s[0:1], 0x0
.Ltmp9911:
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s9, s41, s40
.Ltmp9912:
	s_lshl_b32 s41, s6, 3
.Ltmp9913:
	s_cmp_ge_i32 s41, s9
.Ltmp9914:
	s_cbranch_scc1 .LBB17_115
.Ltmp9915:
	s_load_dwordx4 s[28:31], s[4:5], 0x78
.Ltmp9916:
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
.Ltmp9917:
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
.Ltmp9918:
	s_mul_i32 s0, s6, s0
	s_mul_i32 s1, s6, s28
	s_sub_i32 s0, s7, s0
	s_lshl1_add_u32 s28, s0, s1
.Ltmp9919:
	s_add_u32 s0, s2, s10
	s_addc_u32 s1, s3, s11
	s_sub_i32 s42, s9, s41
	s_load_dword s2, s[0:1], 0x0
.Ltmp9920:
	s_clause 0x2
	s_load_dwordx8 s[20:27], s[4:5], 0x0
	s_load_dwordx2 s[34:35], s[4:5], 0x68
	s_load_dword s11, s[4:5], 0x70
.Ltmp9921:
	s_min_i32 s33, s42, 8
.Ltmp9922:
	s_mov_b32 s0, -1
.Ltmp9923:
	s_waitcnt lgkmcnt(0)
	s_cmp_gt_i32 s2, 0
	s_cbranch_scc0 .LBB17_106
.Ltmp9924:
	v_cmp_gt_u32_e64 s0, 0x200, v0
.Ltmp9925:
	s_and_saveexec_b32 s3, s0
	s_cbranch_execz .LBB17_9
.Ltmp9926:
	v_lshrrev_b32_e32 v1, 5, v0
	v_and_b32_e32 v2, 31, v0
	v_mov_b32_e32 v9, v0
	s_mov_b32 s7, 0
	v_add3_u32 v4, s40, s41, v1
	v_lshlrev_b32_e32 v2, 4, v2
	v_lshlrev_b32_e32 v3, 9, v1
	v_cmp_le_i32_e32 vcc_lo, s42, v1
	v_mov_b32_e32 v1, 0
	v_mad_u64_u32 v[5:6], null, v4, s11, s[28:29]
	v_add_co_u32 v6, s1, s20, v2
	v_add3_u32 v7, 0x80, v3, v2
	v_add_co_ci_u32_e64 v8, null, s21, 0, s1
	s_inst_prefetch 0x1
	s_branch .LBB17_5
.Ltmp9927:
.Ltmp9928:
	.p2align	6
.LBB17_4:
	s_or_b32 exec_lo, exec_lo, s10
.Ltmp9929:
	v_add_nc_u32_e32 v2, 0x100, v9
.Ltmp9930:
	v_cmp_lt_u32_e64 s1, 0xff, v9
	v_mov_b32_e32 v9, v2
.Ltmp9931:
	s_or_b32 s7, s1, s7
	s_andn2_b32 exec_lo, exec_lo, s7
	s_cbranch_execz .LBB17_9
.Ltmp9932:
.LBB17_5:
	v_lshrrev_b32_e32 v2, 8, v9
.Ltmp9933:
	v_lshl_add_u32 v10, v2, 12, v7
.Ltmp9934:
	s_and_saveexec_b32 s1, vcc_lo
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execz .LBB17_7
.Ltmp9935:
	v_mov_b32_e32 v2, v1
.Ltmp9936:
	v_mov_b32_e32 v3, v1
	v_mov_b32_e32 v4, v1
	ds_write_b128 v10, v[1:4]
.Ltmp9937:
.LBB17_7:
	s_andn2_saveexec_b32 s10, s1
	s_cbranch_execz .LBB17_4
.Ltmp9938:
	v_add_lshl_u32 v2, v2, v5, 8
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[2:3], 1, v[2:3]
	v_add_co_u32 v2, s1, v6, v2
	v_add_co_ci_u32_e64 v3, null, v8, v3, s1
.Ltmp9939:
	global_load_dwordx4 v[11:14], v[2:3], off
	s_waitcnt vmcnt(0)
	ds_write_b128 v10, v[11:14]
.Ltmp9940:
	s_branch .LBB17_4
.Ltmp9941:
.LBB17_9:
	s_inst_prefetch 0x2
	s_or_b32 exec_lo, exec_lo, s3
	s_sub_i32 s3, s41, s9
	s_mov_b32 s21, 0
.Ltmp9942:
	s_add_i32 s3, s3, s2
.Ltmp9943:
	s_cmp_gt_i32 s31, 0
	s_mov_b32 s43, 0
	s_cselect_b32 s20, -1, 0
.Ltmp9944:
	s_waitcnt lgkmcnt(0)
.Ltmp9945:
	s_and_b32 vcc_lo, exec_lo, s20
.Ltmp9946:
	s_barrier
	buffer_gl0_inv
	s_cbranch_vccz .LBB17_12
.Ltmp9947:
	s_sub_i32 s1, s3, s31
.Ltmp9948:
	s_cmp_lt_i32 s1, 0
	s_cbranch_scc1 .LBB17_12
.Ltmp9949:
	s_add_i32 s1, s1, 1
.Ltmp9950:
	s_and_b32 s43, s1, -16
.Ltmp9951:
.LBB17_12:
	s_add_i32 s1, s3, s33
	v_and_b32_e32 v33, 15, v0
.Ltmp9952:
	v_bfe_u32 v35, v0, 4, 3
.Ltmp9953:
	s_min_i32 s1, s2, s1
.Ltmp9954:
	s_cmp_eq_u32 s30, 0
	s_cselect_b32 s44, s2, s1
.Ltmp9955:
	v_lshlrev_b32_e32 v34, 4, v33
.Ltmp9956:
	v_cmp_gt_i32_e64 s1, s42, v35
.Ltmp9957:
	s_cmp_ge_i32 s43, s44
	s_cbranch_scc1 .LBB17_102
.Ltmp9958:
	s_clause 0x2
	s_load_dwordx4 s[36:39], s[4:5], 0x50
	s_load_dwordx8 s[12:19], s[4:5], 0x30
	s_load_dword s4, s[4:5], 0x60
.Ltmp9959:
	v_mbcnt_lo_u32_b32 v17, -1, 0
.Ltmp9960:
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
.Ltmp9961:
	v_lshrrev_b32_e32 v20, 4, v0
.Ltmp9962:
	v_mad_u32_u24 v58, v48, s3, 0x80
	v_cmp_lt_u32_e64 s3, v23, v19
	v_cndmask_b32_e32 v21, v17, v21, vcc_lo
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s8, s38, s8
.Ltmp9963:
	v_cmp_lt_u32_e32 vcc_lo, v22, v19
	s_ashr_i32 s9, s8, 31
.Ltmp9964:
	v_cndmask_b32_e64 v23, v17, v23, s3
	s_lshl_b64 s[8:9], s[8:9], 2
	v_mul_lo_u32 v18, v18, s37
	s_add_u32 s26, s26, s8
	s_addc_u32 s27, s27, s9
	s_cmp_lg_u32 s16, 1
	s_mul_i32 s8, s6, s13
	s_cselect_b32 s2, -1, 0
.Ltmp9965:
	s_cmp_lg_u32 s36, 1
	s_mul_i32 s6, s6, s18
.Ltmp9966:
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
.Ltmp9967:
	v_mul_lo_u32 v64, s14, v20
	s_add_u32 s23, s24, s6
	s_addc_u32 s24, s25, s7
	s_cmp_lg_u32 s30, 0
	v_and_b32_e32 v3, 0xf0, v0
	s_cselect_b32 s25, -1, 0
	s_abs_i32 s5, s4
	s_abs_i32 s30, s39
.Ltmp9968:
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
.Ltmp9969:
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
	s_branch .LBB17_15
.Ltmp9970:
.LBB17_14:
	s_or_b32 exec_lo, exec_lo, s3
.Ltmp9971:
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v18, v18, v30
.Ltmp9972:
	s_add_i32 s43, s43, 16
	s_cmp_ge_i32 s43, s44
.Ltmp9973:
	s_barrier
.Ltmp9974:
	v_fmac_f32_e32 v18, v45, v29
.Ltmp9975:
	buffer_gl0_inv
	v_mov_b32_e32 v45, v18
.Ltmp9976:
	s_cbranch_scc1 .LBB17_103
.Ltmp9977:
.LBB17_15:
	s_sub_i32 s37, s44, s43
.Ltmp9978:
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB17_21
.Ltmp9979:
	s_mov_b32 s4, exec_lo
	v_cmpx_le_i32_e64 s37, v0
	s_xor_b32 s4, exec_lo, s4
.Ltmp9980:
	ds_write_b32 v36, v1
.Ltmp9981:
	s_or_saveexec_b32 s4, s4
	v_mov_b32_e32 v18, 0
	s_xor_b32 exec_lo, exec_lo, s4
	s_cbranch_execz .LBB17_20
.Ltmp9982:
	v_or_b32_e32 v18, s43, v0
.Ltmp9983:
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
.Ltmp9984:
	global_load_dword v30, v[30:31], off
	s_waitcnt vmcnt(0)
	ds_write_b32 v36, v30
.Ltmp9985:
.LBB17_20:
	s_or_b32 exec_lo, exec_lo, s4
	ds_write_b32 v36, v18 offset:64
.Ltmp9986:
.LBB17_21:
	s_or_b32 exec_lo, exec_lo, s3
.Ltmp9987:
	s_min_i32 s9, s37, 16
.Ltmp9988:
	s_andn2_b32 vcc_lo, exec_lo, s13
	s_mov_b32 s3, -1
.Ltmp9989:
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
.Ltmp9990:
	s_cbranch_vccnz .LBB17_33
.Ltmp9991:
	v_mov_b32_e32 v30, v0
.Ltmp9992:
	s_and_saveexec_b32 s3, s21
	s_cbranch_execz .LBB17_26
.Ltmp9993:
	s_cmp_gt_i32 s9, 0
	s_cselect_b32 s5, -1, 0
	s_and_saveexec_b32 s4, s5
	s_cbranch_execz .LBB17_25
.Ltmp9994:
	ds_read2_b32 v[29:30], v1 offset1:16
	v_lshlrev_b64 v[76:77], 1, v[21:22]
	v_lshlrev_b64 v[78:79], 1, v[23:24]
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v31, v29, s12
	v_mul_lo_u32 v29, v29, s17
	v_mul_lo_u32 v72, v30, s15
.Ltmp9995:
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
.Ltmp9996:
	global_load_ushort v18, v[29:30], off
.Ltmp9997:
	global_load_ushort v29, v[31:32], off
.Ltmp9998:
	s_waitcnt vmcnt(1)
	ds_write_b16 v40, v18 offset:8192
	s_waitcnt vmcnt(0)
	ds_write_b16 v40, v29 offset:16640
.Ltmp9999:
.LBB17_25:
	s_or_b32 exec_lo, exec_lo, s4
	v_mov_b32_e32 v30, v41
.Ltmp10000:
.LBB17_26:
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v18, 0x100, v30
	v_lshrrev_b32_e32 v29, 8, v30
	v_or_b32_e32 v30, 0xfffffe00, v30
	s_mov_b32 s3, 0
	v_lshrrev_b32_e32 v18, 8, v18
	v_lshlrev_b32_e32 v32, 2, v29
	v_lshlrev_b32_e32 v31, 2, v18
	s_branch .LBB17_28
.Ltmp10001:
.LBB17_27:
	s_or_b32 exec_lo, exec_lo, s4
.Ltmp10002:
	v_add_nc_u32_e32 v30, 0x200, v30
.Ltmp10003:
	v_add_nc_u32_e32 v18, 2, v18
	v_add_nc_u32_e32 v31, 8, v31
	v_add_nc_u32_e32 v29, 2, v29
	v_add_nc_u32_e32 v32, 8, v32
	v_cmp_lt_u32_e32 vcc_lo, 0xdff, v30
.Ltmp10004:
	s_or_b32 s3, vcc_lo, s3
	s_andn2_b32 exec_lo, exec_lo, s3
	s_cbranch_execz .LBB17_32
.Ltmp10005:
.LBB17_28:
	s_mov_b32 s4, exec_lo
	v_cmpx_gt_i32_e64 s9, v29
	s_cbranch_execz .LBB17_30
.Ltmp10006:
	ds_read2_b32 v[72:73], v32 offset1:16
	v_lshlrev_b64 v[80:81], 1, v[21:22]
	v_lshlrev_b64 v[82:83], 1, v[23:24]
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v74, v72, s12
	v_mul_lo_u32 v72, v72, s17
	v_mul_lo_u32 v76, v73, s15
.Ltmp10007:
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
.Ltmp10008:
	global_load_ushort v72, v[72:73], off
.Ltmp10009:
	global_load_ushort v73, v[74:75], off
.Ltmp10010:
	v_mad_u32_u24 v74, 0x108, v29, v0
	v_lshl_add_u32 v74, v74, 1, 0x80
	s_waitcnt vmcnt(1)
	ds_write_b16 v74, v72 offset:8192
	s_waitcnt vmcnt(0)
	ds_write_b16 v74, v73 offset:16640
.Ltmp10011:
.LBB17_30:
	s_or_b32 exec_lo, exec_lo, s4
.Ltmp10012:
	s_mov_b32 s4, exec_lo
	v_cmpx_gt_i32_e64 s9, v18
	s_cbranch_execz .LBB17_27
.Ltmp10013:
	ds_read2_b32 v[72:73], v31 offset1:16
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v74, v72, s12
	v_mul_lo_u32 v72, v72, s17
	v_mul_lo_u32 v76, v73, s15
.Ltmp10014:
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
.Ltmp10015:
	global_load_ushort v72, v[72:73], off
.Ltmp10016:
	global_load_ushort v73, v[74:75], off
.Ltmp10017:
	v_mad_u32_u24 v74, 0x108, v18, v0
	v_lshl_add_u32 v74, v74, 1, 0x80
	s_waitcnt vmcnt(1)
	ds_write_b16 v74, v72 offset:8192
	s_waitcnt vmcnt(0)
	ds_write_b16 v74, v73 offset:16640
.Ltmp10018:
	s_branch .LBB17_27
.Ltmp10019:
.LBB17_32:
	s_or_b32 exec_lo, exec_lo, s3
	s_mov_b32 s3, 0
.Ltmp10020:
.LBB17_33:
	s_and_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccz .LBB17_61
.Ltmp10021:
	s_and_saveexec_b32 s38, s0
	s_cbranch_execz .LBB17_60
.Ltmp10022:
	v_mov_b32_e32 v29, v64
	v_mov_b32_e32 v18, v0
	s_mov_b32 s4, 0
	v_cmp_gt_i32_e32 vcc_lo, s37, v33
	s_inst_prefetch 0x1
	s_branch .LBB17_37
.Ltmp10023:
	.p2align	6
.LBB17_36:
	s_or_b32 exec_lo, exec_lo, s5
.Ltmp10024:
	v_add_nc_u32_e32 v30, 0x100, v18
.Ltmp10025:
	v_cmp_lt_u32_e64 s3, 0xff, v18
	v_add_nc_u32_e32 v29, s14, v29
	v_mov_b32_e32 v18, v30
.Ltmp10026:
	s_or_b32 s4, s3, s4
	s_andn2_b32 exec_lo, exec_lo, s4
	s_cbranch_execz .LBB17_39
.Ltmp10027:
.LBB17_37:
	s_and_saveexec_b32 s5, vcc_lo
	s_cbranch_execz .LBB17_36
.Ltmp10028:
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
.Ltmp10029:
	global_load_dwordx4 v[72:75], v[30:31], off
	v_and_b32_e32 v30, 0x1f0, v18
	v_add_nc_u32_e32 v30, v43, v30
	s_waitcnt vmcnt(0)
	ds_write_b128 v30, v[72:75] offset:8192
	s_branch .LBB17_36
.Ltmp10030:
.LBB17_39:
	s_inst_prefetch 0x2
	s_or_b32 exec_lo, exec_lo, s4
	v_cmp_gt_i32_e64 s3, s9, v50
	v_cmp_gt_i32_e64 s4, s9, v51
	v_cmp_gt_i32_e64 s5, s9, v52
	v_cmp_gt_i32_e64 s6, s9, v53
	v_cmp_gt_i32_e64 s7, s9, v55
	v_cmp_gt_i32_e64 s8, s9, v56
	v_cmp_gt_i32_e64 s9, s9, v57
.Ltmp10031:
	v_mov_b32_e32 v18, v0
	s_mov_b32 s45, 0
	v_cmp_gt_i32_e32 vcc_lo, s37, v48
	s_branch .LBB17_41
.Ltmp10032:
.LBB17_40:
	s_or_b32 exec_lo, exec_lo, s46
.Ltmp10033:
	v_add_nc_u32_e32 v29, 0x100, v18
.Ltmp10034:
	v_cmp_lt_u32_e64 s10, 0xff, v18
	v_mov_b32_e32 v18, v29
.Ltmp10035:
	s_or_b32 s45, s10, s45
	s_andn2_b32 exec_lo, exec_lo, s45
	s_cbranch_execz .LBB17_60
.Ltmp10036:
.LBB17_41:
	s_and_saveexec_b32 s46, vcc_lo
	s_cbranch_execz .LBB17_40
.Ltmp10037:
	ds_read_b32 v29, v59
.Ltmp10038:
	v_lshrrev_b32_e32 v30, 4, v18
.Ltmp10039:
	v_lshrrev_b32_e32 v72, 1, v18
.Ltmp10040:
	s_mov_b32 s47, exec_lo
.Ltmp10041:
	v_mul_lo_u32 v30, v30, s19
	v_ashrrev_i32_e32 v31, 31, v30
	v_lshlrev_b64 v[31:32], 1, v[30:31]
.Ltmp10042:
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v73, 7, v29
	v_ashrrev_i32_e32 v30, 31, v29
	v_cmpx_ne_u32_e32 0, v73
	s_xor_b32 s47, exec_lo, s47
	s_cbranch_execz .LBB17_58
.Ltmp10043:
	ds_read_b32 v73, v49
.Ltmp10044:
	v_add_co_u32 v31, s10, v65, v31
	v_add_co_ci_u32_e64 v32, null, v66, v32, s10
.Ltmp10045:
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
.Ltmp10046:
	v_lshl_add_u32 v29, v72, 1, 0x80
.Ltmp10047:
	v_add_nc_u32_e32 v29, v29, v54
	s_waitcnt vmcnt(0)
	ds_write_b16 v29, v30 offset:16640
.Ltmp10048:
	s_and_saveexec_b32 s48, s3
	s_cbranch_execz .LBB17_45
.Ltmp10049:
	ds_read2_b32 v[72:73], v49 offset0:1 offset1:17
.Ltmp10050:
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
.Ltmp10051:
.LBB17_45:
	s_or_b32 exec_lo, exec_lo, s48
.Ltmp10052:
	s_and_saveexec_b32 s48, s4
	s_cbranch_execz .LBB17_47
.Ltmp10053:
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
.Ltmp10054:
.LBB17_47:
	s_or_b32 exec_lo, exec_lo, s48
.Ltmp10055:
	s_and_saveexec_b32 s48, s5
	s_cbranch_execz .LBB17_49
.Ltmp10056:
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
.Ltmp10057:
.LBB17_49:
	s_or_b32 exec_lo, exec_lo, s48
.Ltmp10058:
	s_and_saveexec_b32 s48, s6
	s_cbranch_execz .LBB17_51
.Ltmp10059:
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
.Ltmp10060:
.LBB17_51:
	s_or_b32 exec_lo, exec_lo, s48
.Ltmp10061:
	s_and_saveexec_b32 s48, s7
	s_cbranch_execz .LBB17_53
.Ltmp10062:
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
.Ltmp10063:
.LBB17_53:
	s_or_b32 exec_lo, exec_lo, s48
.Ltmp10064:
	s_and_saveexec_b32 s48, s8
	s_cbranch_execz .LBB17_55
.Ltmp10065:
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
.Ltmp10066:
.LBB17_55:
	s_or_b32 exec_lo, exec_lo, s48
.Ltmp10067:
	s_and_saveexec_b32 s48, s9
	s_cbranch_execz .LBB17_57
.Ltmp10068:
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
.Ltmp10069:
.LBB17_57:
	s_or_b32 exec_lo, exec_lo, s48
.Ltmp10070:
.LBB17_58:
	s_andn2_saveexec_b32 s10, s47
	s_cbranch_execz .LBB17_40
.Ltmp10071:
	ds_read_b32 v73, v49
.Ltmp10072:
	v_lshlrev_b64 v[29:30], 1, v[29:30]
.Ltmp10073:
	v_lshl_add_u32 v72, v72, 1, v58
.Ltmp10074:
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v73, v73, s17
	v_ashrrev_i32_e32 v74, 31, v73
	v_lshlrev_b64 v[73:74], 1, v[73:74]
	v_add_co_u32 v73, s10, s23, v73
	v_add_co_ci_u32_e64 v74, null, s24, v74, s10
.Ltmp10075:
	v_add_co_u32 v31, s10, v73, v31
	v_add_co_ci_u32_e64 v32, null, v74, v32, s10
.Ltmp10076:
	v_add_co_u32 v31, s10, v31, v19
	v_add_co_ci_u32_e64 v32, null, v32, v20, s10
	v_add_co_u32 v29, s10, v31, v29
	v_add_co_ci_u32_e64 v30, null, v32, v30, s10
.Ltmp10077:
	global_load_dwordx4 v[29:32], v[29:30], off
.Ltmp10078:
	s_waitcnt vmcnt(0)
	ds_write_b16 v72, v29 offset:16640
.Ltmp10079:
	ds_write_b16_d16_hi v72, v29 offset:17168
.Ltmp10080:
	ds_write_b16 v72, v30 offset:17696
.Ltmp10081:
	ds_write_b16_d16_hi v72, v30 offset:18224
.Ltmp10082:
	ds_write_b16 v72, v31 offset:18752
.Ltmp10083:
	ds_write_b16_d16_hi v72, v31 offset:19280
.Ltmp10084:
	ds_write_b16 v72, v32 offset:19808
.Ltmp10085:
	ds_write_b16_d16_hi v72, v32 offset:20336
.Ltmp10086:
	s_branch .LBB17_40
.Ltmp10087:
.LBB17_60:
	s_or_b32 exec_lo, exec_lo, s38
.Ltmp10088:
.LBB17_61:
	v_cmp_gt_i32_e32 vcc_lo, s37, v33
	v_mov_b32_e32 v18, 0xff800000
.Ltmp10089:
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
.Ltmp10090:
	s_and_b32 s3, s1, vcc_lo
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB17_65
.Ltmp10091:
	v_or_b32_e32 v18, s43, v33
.Ltmp10092:
	v_sub_nc_u32_e32 v29, v37, v18
	v_cmp_lt_i32_e32 vcc_lo, v37, v18
	v_mov_b32_e32 v18, 0xff800000
.Ltmp10093:
	v_cmp_le_i32_e64 s3, s31, v29
	s_and_b32 s5, s25, vcc_lo
	s_and_b32 s3, s20, s3
	s_nor_b32 s5, s5, s3
.Ltmp10094:
	s_and_saveexec_b32 s3, s5
	s_cbranch_execz .LBB17_64
.Ltmp10095:
	ds_read_b128 v[29:32], v43 offset:8192
.Ltmp10096:
	ds_read_b128 v[72:75], v44
.Ltmp10097:
	ds_read_b128 v[76:79], v44 offset:16
.Ltmp10098:
	ds_read_b128 v[80:83], v43 offset:8208
.Ltmp10099:
	v_mov_b32_e32 v18, 0
.Ltmp10100:
	v_mov_b32_e32 v92, 0
.Ltmp10101:
	ds_read_b128 v[84:87], v44 offset:32
.Ltmp10102:
	ds_read_b128 v[88:91], v43 offset:8224
.Ltmp10103:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v72, v29
.Ltmp10104:
	v_dot2c_f32_f16 v92, v73, v30
.Ltmp10105:
	v_dot2c_f32_f16 v18, v74, v31
.Ltmp10106:
	v_dot2c_f32_f16 v92, v75, v32
.Ltmp10107:
	ds_read_b128 v[29:32], v44 offset:48
.Ltmp10108:
	ds_read_b128 v[72:75], v43 offset:8240
.Ltmp10109:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v76, v80
.Ltmp10110:
	v_dot2c_f32_f16 v92, v77, v81
.Ltmp10111:
	v_dot2c_f32_f16 v18, v78, v82
.Ltmp10112:
	v_dot2c_f32_f16 v92, v79, v83
.Ltmp10113:
	ds_read_b128 v[76:79], v44 offset:64
.Ltmp10114:
	ds_read_b128 v[80:83], v43 offset:8256
.Ltmp10115:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v84, v88
.Ltmp10116:
	v_dot2c_f32_f16 v92, v85, v89
.Ltmp10117:
	v_dot2c_f32_f16 v18, v86, v90
.Ltmp10118:
	v_dot2c_f32_f16 v92, v87, v91
.Ltmp10119:
	ds_read_b128 v[84:87], v44 offset:80
.Ltmp10120:
	ds_read_b128 v[88:91], v43 offset:8272
.Ltmp10121:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v29, v72
.Ltmp10122:
	v_dot2c_f32_f16 v92, v30, v73
.Ltmp10123:
	v_dot2c_f32_f16 v18, v31, v74
.Ltmp10124:
	v_dot2c_f32_f16 v92, v32, v75
.Ltmp10125:
	ds_read_b128 v[29:32], v44 offset:96
.Ltmp10126:
	ds_read_b128 v[72:75], v43 offset:8288
.Ltmp10127:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v76, v80
.Ltmp10128:
	v_dot2c_f32_f16 v92, v77, v81
.Ltmp10129:
	v_dot2c_f32_f16 v18, v78, v82
.Ltmp10130:
	v_dot2c_f32_f16 v92, v79, v83
.Ltmp10131:
	ds_read_b128 v[76:79], v44 offset:112
.Ltmp10132:
	ds_read_b128 v[80:83], v43 offset:8304
.Ltmp10133:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v84, v88
.Ltmp10134:
	v_dot2c_f32_f16 v92, v85, v89
.Ltmp10135:
	v_dot2c_f32_f16 v18, v86, v90
.Ltmp10136:
	v_dot2c_f32_f16 v92, v87, v91
.Ltmp10137:
	ds_read_b128 v[84:87], v44 offset:128
.Ltmp10138:
	ds_read_b128 v[88:91], v43 offset:8320
.Ltmp10139:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v29, v72
.Ltmp10140:
	v_dot2c_f32_f16 v92, v30, v73
.Ltmp10141:
	v_dot2c_f32_f16 v18, v31, v74
.Ltmp10142:
	v_dot2c_f32_f16 v92, v32, v75
.Ltmp10143:
	ds_read_b128 v[29:32], v44 offset:144
.Ltmp10144:
	ds_read_b128 v[72:75], v43 offset:8336
.Ltmp10145:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v76, v80
.Ltmp10146:
	v_dot2c_f32_f16 v92, v77, v81
.Ltmp10147:
	v_dot2c_f32_f16 v18, v78, v82
.Ltmp10148:
	v_dot2c_f32_f16 v92, v79, v83
.Ltmp10149:
	ds_read_b128 v[76:79], v44 offset:160
.Ltmp10150:
	ds_read_b128 v[80:83], v43 offset:8352
.Ltmp10151:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v84, v88
.Ltmp10152:
	v_dot2c_f32_f16 v92, v85, v89
.Ltmp10153:
	v_dot2c_f32_f16 v18, v86, v90
.Ltmp10154:
	v_dot2c_f32_f16 v92, v87, v91
.Ltmp10155:
	ds_read_b128 v[84:87], v44 offset:176
.Ltmp10156:
	ds_read_b128 v[88:91], v43 offset:8368
.Ltmp10157:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v29, v72
.Ltmp10158:
	v_dot2c_f32_f16 v92, v30, v73
.Ltmp10159:
	v_dot2c_f32_f16 v18, v31, v74
.Ltmp10160:
	v_dot2c_f32_f16 v92, v32, v75
.Ltmp10161:
	ds_read_b128 v[29:32], v44 offset:192
.Ltmp10162:
	ds_read_b128 v[72:75], v43 offset:8384
.Ltmp10163:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v76, v80
.Ltmp10164:
	v_dot2c_f32_f16 v92, v77, v81
.Ltmp10165:
	v_dot2c_f32_f16 v18, v78, v82
.Ltmp10166:
	v_dot2c_f32_f16 v92, v79, v83
.Ltmp10167:
	ds_read_b128 v[76:79], v44 offset:208
.Ltmp10168:
	ds_read_b128 v[80:83], v43 offset:8400
.Ltmp10169:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v84, v88
.Ltmp10170:
	v_dot2c_f32_f16 v92, v85, v89
.Ltmp10171:
	v_dot2c_f32_f16 v18, v86, v90
.Ltmp10172:
	v_dot2c_f32_f16 v92, v87, v91
.Ltmp10173:
	ds_read_b128 v[84:87], v44 offset:224
.Ltmp10174:
	ds_read_b128 v[88:91], v43 offset:8416
.Ltmp10175:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v29, v72
.Ltmp10176:
	v_dot2c_f32_f16 v92, v30, v73
.Ltmp10177:
	v_dot2c_f32_f16 v18, v31, v74
.Ltmp10178:
	v_dot2c_f32_f16 v92, v32, v75
.Ltmp10179:
	ds_read_b128 v[29:32], v44 offset:240
.Ltmp10180:
	ds_read_b128 v[72:75], v43 offset:8432
.Ltmp10181:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v76, v80
.Ltmp10182:
	v_dot2c_f32_f16 v92, v77, v81
.Ltmp10183:
	v_dot2c_f32_f16 v18, v78, v82
.Ltmp10184:
	v_dot2c_f32_f16 v92, v79, v83
.Ltmp10185:
	ds_read_b128 v[76:79], v44 offset:256
.Ltmp10186:
	ds_read_b128 v[80:83], v43 offset:8448
.Ltmp10187:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v84, v88
.Ltmp10188:
	v_dot2c_f32_f16 v92, v85, v89
.Ltmp10189:
	v_dot2c_f32_f16 v18, v86, v90
.Ltmp10190:
	v_dot2c_f32_f16 v92, v87, v91
.Ltmp10191:
	ds_read_b128 v[84:87], v44 offset:272
.Ltmp10192:
	ds_read_b128 v[88:91], v43 offset:8464
.Ltmp10193:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v29, v72
.Ltmp10194:
	v_dot2c_f32_f16 v92, v30, v73
.Ltmp10195:
	v_dot2c_f32_f16 v18, v31, v74
.Ltmp10196:
	v_dot2c_f32_f16 v92, v32, v75
.Ltmp10197:
	ds_read_b128 v[29:32], v44 offset:288
.Ltmp10198:
	ds_read_b128 v[72:75], v43 offset:8480
.Ltmp10199:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v76, v80
.Ltmp10200:
	v_dot2c_f32_f16 v92, v77, v81
.Ltmp10201:
	v_dot2c_f32_f16 v18, v78, v82
.Ltmp10202:
	v_dot2c_f32_f16 v92, v79, v83
.Ltmp10203:
	ds_read_b128 v[76:79], v44 offset:304
.Ltmp10204:
	ds_read_b128 v[80:83], v43 offset:8496
.Ltmp10205:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v84, v88
.Ltmp10206:
	v_dot2c_f32_f16 v92, v85, v89
.Ltmp10207:
	v_dot2c_f32_f16 v18, v86, v90
.Ltmp10208:
	v_dot2c_f32_f16 v92, v87, v91
.Ltmp10209:
	ds_read_b128 v[84:87], v44 offset:320
.Ltmp10210:
	ds_read_b128 v[88:91], v43 offset:8512
.Ltmp10211:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v29, v72
.Ltmp10212:
	v_dot2c_f32_f16 v92, v30, v73
.Ltmp10213:
	v_dot2c_f32_f16 v18, v31, v74
.Ltmp10214:
	v_dot2c_f32_f16 v92, v32, v75
.Ltmp10215:
	ds_read_b128 v[29:32], v44 offset:336
.Ltmp10216:
	ds_read_b128 v[72:75], v43 offset:8528
.Ltmp10217:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v76, v80
.Ltmp10218:
	v_dot2c_f32_f16 v92, v77, v81
.Ltmp10219:
	v_dot2c_f32_f16 v18, v78, v82
.Ltmp10220:
	v_dot2c_f32_f16 v92, v79, v83
.Ltmp10221:
	ds_read_b128 v[76:79], v44 offset:352
.Ltmp10222:
	ds_read_b128 v[80:83], v43 offset:8544
.Ltmp10223:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v84, v88
.Ltmp10224:
	v_dot2c_f32_f16 v92, v85, v89
.Ltmp10225:
	v_dot2c_f32_f16 v18, v86, v90
.Ltmp10226:
	v_dot2c_f32_f16 v92, v87, v91
.Ltmp10227:
	ds_read_b128 v[84:87], v44 offset:368
.Ltmp10228:
	ds_read_b128 v[88:91], v43 offset:8560
.Ltmp10229:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v29, v72
.Ltmp10230:
	v_dot2c_f32_f16 v92, v30, v73
.Ltmp10231:
	v_dot2c_f32_f16 v18, v31, v74
.Ltmp10232:
	v_dot2c_f32_f16 v92, v32, v75
.Ltmp10233:
	ds_read_b128 v[29:32], v44 offset:384
.Ltmp10234:
	ds_read_b128 v[72:75], v43 offset:8576
.Ltmp10235:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v76, v80
.Ltmp10236:
	v_dot2c_f32_f16 v92, v77, v81
.Ltmp10237:
	v_dot2c_f32_f16 v18, v78, v82
.Ltmp10238:
	v_dot2c_f32_f16 v92, v79, v83
.Ltmp10239:
	ds_read_b128 v[76:79], v44 offset:400
.Ltmp10240:
	ds_read_b128 v[80:83], v43 offset:8592
.Ltmp10241:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v84, v88
.Ltmp10242:
	v_dot2c_f32_f16 v92, v85, v89
.Ltmp10243:
	v_dot2c_f32_f16 v18, v86, v90
.Ltmp10244:
	v_dot2c_f32_f16 v92, v87, v91
.Ltmp10245:
	ds_read_b128 v[84:87], v44 offset:416
.Ltmp10246:
	ds_read_b128 v[88:91], v43 offset:8608
.Ltmp10247:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v29, v72
.Ltmp10248:
	v_dot2c_f32_f16 v92, v30, v73
.Ltmp10249:
	v_dot2c_f32_f16 v18, v31, v74
.Ltmp10250:
	v_dot2c_f32_f16 v92, v32, v75
.Ltmp10251:
	ds_read_b128 v[29:32], v44 offset:432
.Ltmp10252:
	ds_read_b128 v[72:75], v43 offset:8624
.Ltmp10253:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v76, v80
.Ltmp10254:
	v_dot2c_f32_f16 v92, v77, v81
.Ltmp10255:
	v_dot2c_f32_f16 v18, v78, v82
.Ltmp10256:
	v_dot2c_f32_f16 v92, v79, v83
.Ltmp10257:
	ds_read_b128 v[76:79], v44 offset:448
.Ltmp10258:
	ds_read_b128 v[80:83], v43 offset:8640
.Ltmp10259:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v84, v88
.Ltmp10260:
	v_dot2c_f32_f16 v92, v85, v89
.Ltmp10261:
	v_dot2c_f32_f16 v18, v86, v90
.Ltmp10262:
	v_dot2c_f32_f16 v92, v87, v91
.Ltmp10263:
	ds_read_b128 v[84:87], v44 offset:464
.Ltmp10264:
	ds_read_b128 v[88:91], v43 offset:8656
.Ltmp10265:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v29, v72
.Ltmp10266:
	v_dot2c_f32_f16 v92, v30, v73
.Ltmp10267:
	v_dot2c_f32_f16 v18, v31, v74
.Ltmp10268:
	v_dot2c_f32_f16 v92, v32, v75
.Ltmp10269:
	ds_read_b128 v[29:32], v44 offset:480
.Ltmp10270:
	ds_read_b128 v[72:75], v43 offset:8672
.Ltmp10271:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v76, v80
.Ltmp10272:
	v_dot2c_f32_f16 v92, v77, v81
.Ltmp10273:
	v_dot2c_f32_f16 v18, v78, v82
.Ltmp10274:
	v_dot2c_f32_f16 v92, v79, v83
.Ltmp10275:
	ds_read_b128 v[76:79], v44 offset:496
.Ltmp10276:
	ds_read_b128 v[80:83], v43 offset:8688
.Ltmp10277:
	s_waitcnt lgkmcnt(4)
	v_dot2c_f32_f16 v18, v84, v88
.Ltmp10278:
	v_dot2c_f32_f16 v92, v85, v89
.Ltmp10279:
	v_dot2c_f32_f16 v18, v86, v90
.Ltmp10280:
	v_dot2c_f32_f16 v92, v87, v91
.Ltmp10281:
	s_waitcnt lgkmcnt(2)
	v_dot2c_f32_f16 v18, v29, v72
.Ltmp10282:
	v_dot2c_f32_f16 v92, v30, v73
.Ltmp10283:
	v_dot2c_f32_f16 v18, v31, v74
.Ltmp10284:
	v_dot2c_f32_f16 v92, v32, v75
.Ltmp10285:
	s_waitcnt lgkmcnt(0)
	v_dot2c_f32_f16 v18, v76, v80
.Ltmp10286:
	v_dot2c_f32_f16 v92, v77, v81
.Ltmp10287:
	v_dot2c_f32_f16 v18, v78, v82
.Ltmp10288:
	v_dot2c_f32_f16 v92, v79, v83
.Ltmp10289:
	v_add_f32_e32 v18, v92, v18
.Ltmp10290:
	v_mul_f32_e32 v18, s29, v18
.Ltmp10291:
.LBB17_64:
	s_or_b32 exec_lo, exec_lo, s3
.Ltmp10292:
.LBB17_65:
	s_or_b32 exec_lo, exec_lo, s4
.Ltmp10293:
	ds_bpermute_b32 v29, v60, v18
.Ltmp10294:
	v_max_f32_e32 v30, v18, v18
	v_mov_b32_e32 v31, 0
.Ltmp10295:
	s_mov_b32 s3, exec_lo
.Ltmp10296:
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v29, v29, v29
.Ltmp10297:
	v_max_f32_e32 v29, v30, v29
.Ltmp10298:
	ds_bpermute_b32 v30, v61, v29
.Ltmp10299:
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v30, v30, v30
.Ltmp10300:
	v_max_f32_e32 v29, v29, v30
.Ltmp10301:
	ds_bpermute_b32 v30, v62, v29
.Ltmp10302:
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v30, v30, v30
.Ltmp10303:
	v_max_f32_e32 v29, v29, v30
.Ltmp10304:
	ds_bpermute_b32 v30, v63, v29
.Ltmp10305:
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v30, v30, v30
.Ltmp10306:
	v_max_f32_e32 v30, v29, v30
.Ltmp10307:
	v_mov_b32_e32 v29, 1.0
.Ltmp10308:
	v_cmpx_lg_f32_e32 0xff800000, v30
	s_cbranch_execz .LBB17_69
.Ltmp10309:
	v_max_f32_e32 v29, v30, v30
	v_max_f32_e32 v30, v42, v42
.Ltmp10310:
	s_mov_b32 s4, exec_lo
.Ltmp10311:
	v_max_f32_e32 v30, v30, v29
.Ltmp10312:
	v_mov_b32_e32 v29, 0
.Ltmp10313:
	v_cmpx_neq_f32_e32 0xff800000, v42
	s_cbranch_execz .LBB17_68
.Ltmp10314:
	v_sub_f32_e32 v29, v42, v30
.Ltmp10315:
	v_mul_f32_e32 v31, 0x3fb8aa3b, v29
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v29
	v_fma_f32 v32, 0x3fb8aa3b, v29, -v31
	v_rndne_f32_e32 v42, v31
.Ltmp10316:
	v_fmac_f32_e32 v32, 0x32a5705f, v29
	v_sub_f32_e32 v31, v31, v42
	v_add_f32_e32 v31, v31, v32
	v_cvt_i32_f32_e32 v32, v42
	v_exp_f32_e32 v31, v31
	v_ldexp_f32 v31, v31, v32
	v_cndmask_b32_e32 v31, 0, v31, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v29
	v_cndmask_b32_e32 v29, 0x7f800000, v31, vcc_lo
.Ltmp10317:
.LBB17_68:
	s_or_b32 exec_lo, exec_lo, s4
.Ltmp10318:
	v_sub_f32_e32 v18, v18, v30
.Ltmp10319:
	v_mov_b32_e32 v42, v30
.Ltmp10320:
	v_mul_f32_e32 v18, 0x3fb8aa3b, v18
.Ltmp10321:
	v_exp_f32_e32 v31, v18
.Ltmp10322:
.LBB17_69:
	s_or_b32 exec_lo, exec_lo, s3
.Ltmp10323:
	ds_bpermute_b32 v18, v60, v31
.Ltmp10324:
	v_mul_f32_e32 v2, v29, v2
.Ltmp10325:
	v_mul_f32_e32 v3, v29, v3
.Ltmp10326:
	v_mul_f32_e32 v4, v29, v4
.Ltmp10327:
	v_mul_f32_e32 v5, v29, v5
.Ltmp10328:
	v_mul_f32_e32 v6, v29, v6
.Ltmp10329:
	v_mul_f32_e32 v7, v29, v7
.Ltmp10330:
	v_mul_f32_e32 v8, v29, v8
.Ltmp10331:
	v_mul_f32_e32 v9, v29, v9
.Ltmp10332:
	v_mul_f32_e32 v10, v29, v10
.Ltmp10333:
	v_mul_f32_e32 v11, v29, v11
.Ltmp10334:
	v_mul_f32_e32 v12, v29, v12
.Ltmp10335:
	v_mul_f32_e32 v13, v29, v13
.Ltmp10336:
	v_mul_f32_e32 v14, v29, v14
.Ltmp10337:
	v_mul_f32_e32 v15, v29, v15
.Ltmp10338:
	v_mul_f32_e32 v16, v29, v16
.Ltmp10339:
	v_mul_f32_e32 v17, v29, v17
.Ltmp10340:
	ds_write_b32 v46, v31 offset:25088
.Ltmp10341:
	s_waitcnt lgkmcnt(0)
	s_barrier
.Ltmp10342:
	v_add_f32_e32 v18, v31, v18
.Ltmp10343:
	buffer_gl0_inv
.Ltmp10344:
	ds_bpermute_b32 v30, v61, v18
.Ltmp10345:
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v18, v18, v30
.Ltmp10346:
	ds_bpermute_b32 v30, v62, v18
.Ltmp10347:
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v18, v18, v30
.Ltmp10348:
	ds_bpermute_b32 v30, v63, v18
.Ltmp10349:
	s_and_saveexec_b32 s3, s1
	s_cbranch_execz .LBB17_14
.Ltmp10350:
	s_cmp_lt_i32 s37, 1
	s_cbranch_scc1 .LBB17_72
.Ltmp10351:
	ds_read_b32 v31, v47 offset:25088
.Ltmp10352:
	ds_read_b128 v[72:75], v38 offset:16640
.Ltmp10353:
	ds_read_b128 v[76:79], v38 offset:16656
.Ltmp10354:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp10355:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10356:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp10357:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10358:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp10359:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10360:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp10361:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10362:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp10363:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10364:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp10365:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10366:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp10367:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10368:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp10369:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10370:
.LBB17_72:
	s_cmp_lt_i32 s37, 2
	s_cbranch_scc0 .LBB17_87
.Ltmp10371:
	s_cmp_lt_i32 s37, 3
	s_cbranch_scc0 .LBB17_88
.Ltmp10372:
.LBB17_74:
	s_cmp_lt_i32 s37, 4
	s_cbranch_scc0 .LBB17_89
.Ltmp10373:
.LBB17_75:
	s_cmp_lt_i32 s37, 5
	s_cbranch_scc0 .LBB17_90
.Ltmp10374:
.LBB17_76:
	s_cmp_lt_i32 s37, 6
	s_cbranch_scc0 .LBB17_91
.Ltmp10375:
.LBB17_77:
	s_cmp_lt_i32 s37, 7
	s_cbranch_scc0 .LBB17_92
.Ltmp10376:
.LBB17_78:
	s_cmp_lt_i32 s37, 8
	s_cbranch_scc0 .LBB17_93
.Ltmp10377:
.LBB17_79:
	s_cmp_lt_i32 s37, 9
	s_cbranch_scc0 .LBB17_94
.Ltmp10378:
.LBB17_80:
	s_cmp_lt_i32 s37, 10
	s_cbranch_scc0 .LBB17_95
.Ltmp10379:
.LBB17_81:
	s_cmp_lt_i32 s37, 11
	s_cbranch_scc0 .LBB17_96
.Ltmp10380:
.LBB17_82:
	s_cmp_lt_i32 s37, 12
	s_cbranch_scc0 .LBB17_97
.Ltmp10381:
.LBB17_83:
	s_cmp_lt_i32 s37, 13
	s_cbranch_scc0 .LBB17_98
.Ltmp10382:
.LBB17_84:
	s_cmp_lt_i32 s37, 14
	s_cbranch_scc0 .LBB17_99
.Ltmp10383:
.LBB17_85:
	s_cmp_lt_i32 s37, 15
	s_cbranch_scc0 .LBB17_100
.Ltmp10384:
.LBB17_86:
	s_cmp_lt_i32 s37, 16
	s_cbranch_scc1 .LBB17_14
	s_branch .LBB17_101
.Ltmp10385:
.LBB17_87:
	ds_read_b32 v31, v47 offset:25092
.Ltmp10386:
	ds_read_b128 v[72:75], v38 offset:17168
.Ltmp10387:
	ds_read_b128 v[76:79], v38 offset:17184
.Ltmp10388:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp10389:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10390:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp10391:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10392:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp10393:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10394:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp10395:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10396:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp10397:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10398:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp10399:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10400:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp10401:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10402:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp10403:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10404:
	s_cmp_lt_i32 s37, 3
	s_cbranch_scc1 .LBB17_74
.Ltmp10405:
.LBB17_88:
	ds_read_b32 v31, v47 offset:25096
.Ltmp10406:
	ds_read_b128 v[72:75], v38 offset:17696
.Ltmp10407:
	ds_read_b128 v[76:79], v38 offset:17712
.Ltmp10408:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp10409:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10410:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp10411:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10412:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp10413:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10414:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp10415:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10416:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp10417:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10418:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp10419:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10420:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp10421:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10422:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp10423:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10424:
	s_cmp_lt_i32 s37, 4
	s_cbranch_scc1 .LBB17_75
.Ltmp10425:
.LBB17_89:
	ds_read_b32 v31, v47 offset:25100
.Ltmp10426:
	ds_read_b128 v[72:75], v38 offset:18224
.Ltmp10427:
	ds_read_b128 v[76:79], v38 offset:18240
.Ltmp10428:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp10429:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10430:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp10431:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10432:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp10433:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10434:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp10435:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10436:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp10437:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10438:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp10439:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10440:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp10441:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10442:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp10443:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10444:
	s_cmp_lt_i32 s37, 5
	s_cbranch_scc1 .LBB17_76
.Ltmp10445:
.LBB17_90:
	ds_read_b32 v31, v47 offset:25104
.Ltmp10446:
	ds_read_b128 v[72:75], v38 offset:18752
.Ltmp10447:
	ds_read_b128 v[76:79], v38 offset:18768
.Ltmp10448:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp10449:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10450:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp10451:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10452:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp10453:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10454:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp10455:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10456:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp10457:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10458:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp10459:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10460:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp10461:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10462:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp10463:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10464:
	s_cmp_lt_i32 s37, 6
	s_cbranch_scc1 .LBB17_77
.Ltmp10465:
.LBB17_91:
	ds_read_b32 v31, v47 offset:25108
.Ltmp10466:
	ds_read_b128 v[72:75], v38 offset:19280
.Ltmp10467:
	ds_read_b128 v[76:79], v38 offset:19296
.Ltmp10468:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp10469:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10470:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp10471:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10472:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp10473:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10474:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp10475:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10476:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp10477:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10478:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp10479:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10480:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp10481:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10482:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp10483:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10484:
	s_cmp_lt_i32 s37, 7
	s_cbranch_scc1 .LBB17_78
.Ltmp10485:
.LBB17_92:
	ds_read_b32 v31, v47 offset:25112
.Ltmp10486:
	ds_read_b128 v[72:75], v38 offset:19808
.Ltmp10487:
	ds_read_b128 v[76:79], v38 offset:19824
.Ltmp10488:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp10489:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10490:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp10491:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10492:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp10493:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10494:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp10495:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10496:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp10497:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10498:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp10499:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10500:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp10501:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10502:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp10503:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10504:
	s_cmp_lt_i32 s37, 8
	s_cbranch_scc1 .LBB17_79
.Ltmp10505:
.LBB17_93:
	ds_read_b32 v31, v47 offset:25116
.Ltmp10506:
	ds_read_b128 v[72:75], v38 offset:20336
.Ltmp10507:
	ds_read_b128 v[76:79], v38 offset:20352
.Ltmp10508:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp10509:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10510:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp10511:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10512:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp10513:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10514:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp10515:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10516:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp10517:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10518:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp10519:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10520:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp10521:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10522:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp10523:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10524:
	s_cmp_lt_i32 s37, 9
	s_cbranch_scc1 .LBB17_80
.Ltmp10525:
.LBB17_94:
	ds_read_b32 v31, v47 offset:25120
.Ltmp10526:
	ds_read_b128 v[72:75], v38 offset:20864
.Ltmp10527:
	ds_read_b128 v[76:79], v38 offset:20880
.Ltmp10528:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp10529:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10530:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp10531:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10532:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp10533:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10534:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp10535:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10536:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp10537:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10538:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp10539:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10540:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp10541:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10542:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp10543:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10544:
	s_cmp_lt_i32 s37, 10
	s_cbranch_scc1 .LBB17_81
.Ltmp10545:
.LBB17_95:
	ds_read_b32 v31, v47 offset:25124
.Ltmp10546:
	ds_read_b128 v[72:75], v38 offset:21392
.Ltmp10547:
	ds_read_b128 v[76:79], v38 offset:21408
.Ltmp10548:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp10549:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10550:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp10551:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10552:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp10553:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10554:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp10555:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10556:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp10557:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10558:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp10559:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10560:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp10561:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10562:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp10563:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10564:
	s_cmp_lt_i32 s37, 11
	s_cbranch_scc1 .LBB17_82
.Ltmp10565:
.LBB17_96:
	ds_read_b32 v31, v47 offset:25128
.Ltmp10566:
	ds_read_b128 v[72:75], v38 offset:21920
.Ltmp10567:
	ds_read_b128 v[76:79], v38 offset:21936
.Ltmp10568:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp10569:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10570:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp10571:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10572:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp10573:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10574:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp10575:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10576:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp10577:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10578:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp10579:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10580:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp10581:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10582:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp10583:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10584:
	s_cmp_lt_i32 s37, 12
	s_cbranch_scc1 .LBB17_83
.Ltmp10585:
.LBB17_97:
	ds_read_b32 v31, v47 offset:25132
.Ltmp10586:
	ds_read_b128 v[72:75], v38 offset:22448
.Ltmp10587:
	ds_read_b128 v[76:79], v38 offset:22464
.Ltmp10588:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp10589:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10590:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp10591:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10592:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp10593:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10594:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp10595:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10596:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp10597:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10598:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp10599:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10600:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp10601:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10602:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp10603:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10604:
	s_cmp_lt_i32 s37, 13
	s_cbranch_scc1 .LBB17_84
.Ltmp10605:
.LBB17_98:
	ds_read_b32 v31, v47 offset:25136
.Ltmp10606:
	ds_read_b128 v[72:75], v38 offset:22976
.Ltmp10607:
	ds_read_b128 v[76:79], v38 offset:22992
.Ltmp10608:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp10609:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10610:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp10611:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10612:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp10613:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10614:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp10615:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10616:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp10617:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10618:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp10619:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10620:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp10621:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10622:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp10623:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10624:
	s_cmp_lt_i32 s37, 14
	s_cbranch_scc1 .LBB17_85
.Ltmp10625:
.LBB17_99:
	ds_read_b32 v31, v47 offset:25140
.Ltmp10626:
	ds_read_b128 v[72:75], v38 offset:23504
.Ltmp10627:
	ds_read_b128 v[76:79], v38 offset:23520
.Ltmp10628:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp10629:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10630:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp10631:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10632:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp10633:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10634:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp10635:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10636:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp10637:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10638:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp10639:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10640:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp10641:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10642:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp10643:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10644:
	s_cmp_lt_i32 s37, 15
	s_cbranch_scc1 .LBB17_86
.Ltmp10645:
.LBB17_100:
	ds_read_b32 v31, v47 offset:25144
.Ltmp10646:
	ds_read_b128 v[72:75], v38 offset:24032
.Ltmp10647:
	ds_read_b128 v[76:79], v38 offset:24048
.Ltmp10648:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp10649:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10650:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp10651:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10652:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp10653:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10654:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp10655:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10656:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp10657:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10658:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp10659:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10660:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp10661:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10662:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp10663:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10664:
	s_cmp_lt_i32 s37, 16
	s_cbranch_scc1 .LBB17_14
.Ltmp10665:
.LBB17_101:
	ds_read_b32 v31, v47 offset:25148
.Ltmp10666:
	ds_read_b128 v[72:75], v38 offset:24560
.Ltmp10667:
	ds_read_b128 v[76:79], v38 offset:24576
.Ltmp10668:
	s_waitcnt lgkmcnt(1)
	v_fma_mix_f32 v2, v31, v72, v2 op_sel_hi:[0,1,0]
.Ltmp10669:
	v_fma_mix_f32 v3, v31, v72, v3 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10670:
	v_fma_mix_f32 v4, v31, v73, v4 op_sel_hi:[0,1,0]
.Ltmp10671:
	v_fma_mix_f32 v5, v31, v73, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10672:
	v_fma_mix_f32 v6, v31, v74, v6 op_sel_hi:[0,1,0]
.Ltmp10673:
	v_fma_mix_f32 v7, v31, v74, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10674:
	v_fma_mix_f32 v8, v31, v75, v8 op_sel_hi:[0,1,0]
.Ltmp10675:
	v_fma_mix_f32 v9, v31, v75, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10676:
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v10, v31, v76, v10 op_sel_hi:[0,1,0]
.Ltmp10677:
	v_fma_mix_f32 v11, v31, v76, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10678:
	v_fma_mix_f32 v12, v31, v77, v12 op_sel_hi:[0,1,0]
.Ltmp10679:
	v_fma_mix_f32 v13, v31, v77, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10680:
	v_fma_mix_f32 v14, v31, v78, v14 op_sel_hi:[0,1,0]
.Ltmp10681:
	v_fma_mix_f32 v15, v31, v78, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10682:
	v_fma_mix_f32 v16, v31, v79, v16 op_sel_hi:[0,1,0]
.Ltmp10683:
	v_fma_mix_f32 v17, v31, v79, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0]
.Ltmp10684:
	s_branch .LBB17_14
.Ltmp10685:
.LBB17_102:
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
.Ltmp10686:
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
.Ltmp10687:
.LBB17_103:
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s42, v35
	s_cbranch_execz .LBB17_105
.Ltmp10688:
	v_div_scale_f32 v1, null, v18, v18, 1.0
	v_or_b32_e32 v19, s41, v35
	v_div_scale_f32 v22, vcc_lo, 1.0, v18, 1.0
	v_rcp_f32_e32 v21, v1
.Ltmp10689:
	v_add_nc_u32_e32 v19, s40, v19
	v_mul_lo_u32 v19, v19, s11
	v_fma_f32 v20, -v1, v21, 1.0
.Ltmp10690:
	v_fmac_f32_e32 v21, v20, v21
	v_lshrrev_b32_e32 v20, 7, v0
.Ltmp10691:
	v_mul_f32_e32 v23, v22, v21
	v_add_nc_u32_e32 v20, s28, v20
.Ltmp10692:
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
.Ltmp10693:
	v_add_co_u32 v26, vcc_lo, v19, v22
	v_add_co_ci_u32_e64 v27, null, 0, v20, vcc_lo
.Ltmp10694:
	v_fma_mixlo_f16 v18, v1, v2, 0
.Ltmp10695:
	v_fma_mixlo_f16 v19, v1, v4, 0
.Ltmp10696:
	v_fma_mixlo_f16 v20, v1, v6, 0
.Ltmp10697:
	v_fma_mixlo_f16 v21, v1, v8, 0
	v_fma_mixlo_f16 v22, v1, v10, 0
.Ltmp10698:
	v_fma_mixlo_f16 v23, v1, v12, 0
.Ltmp10699:
	v_fma_mixlo_f16 v24, v1, v14, 0
.Ltmp10700:
	v_fma_mixlo_f16 v25, v1, v16, 0
.Ltmp10701:
	v_fma_mixhi_f16 v21, v1, v9, 0
	v_fma_mixhi_f16 v20, v1, v7, 0
	v_fma_mixhi_f16 v19, v1, v5, 0
	v_fma_mixhi_f16 v18, v1, v3, 0
	v_fma_mixhi_f16 v25, v1, v17, 0
	v_fma_mixhi_f16 v24, v1, v15, 0
	v_fma_mixhi_f16 v23, v1, v13, 0
	v_fma_mixhi_f16 v22, v1, v11, 0
	global_store_dwordx4 v[26:27], v[18:21], off
.Ltmp10702:
	global_store_dwordx4 v[26:27], v[22:25], off offset:16
.Ltmp10703:
.LBB17_105:
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.Ltmp10704:
.LBB17_106:
	s_and_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccz .LBB17_115
.Ltmp10705:
	s_lshl_b32 s3, s33, 9
.Ltmp10706:
	s_mov_b32 s0, exec_lo
.Ltmp10707:
	v_cmpx_gt_i32_e64 s3, v0
.Ltmp10708:
	s_cbranch_execz .LBB17_115
.Ltmp10709:
	v_xad_u32 v1, v0, -1, s3
	s_lshl_b32 s5, s33, 8
	s_add_i32 s4, s40, s41
	s_mov_b32 s0, -1
	s_mov_b32 s7, 0
	v_lshrrev_b32_e32 v2, 8, v1
	v_mov_b32_e32 v1, v0
	s_mov_b32 s6, exec_lo
	v_cmpx_ne_u32_e32 0, v2
	s_cbranch_execz .LBB17_112
.Ltmp10710:
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
.Ltmp10711:
.LBB17_110:
	v_mul_hi_u32 v10, v5, v7
	v_mul_hi_u32 v13, v4, v7
	v_lshrrev_b32_e32 v11, 8, v5
	v_lshrrev_b32_e32 v12, 8, v4
.Ltmp10712:
	v_add_nc_u32_e32 v9, -2, v9
.Ltmp10713:
	v_mul_hi_u32 v14, v11, v8
	v_mul_lo_u32 v16, v10, s8
	v_mul_lo_u32 v17, v13, s8
	v_mul_hi_u32 v15, v12, v8
	v_add_nc_u32_e32 v18, 1, v10
	v_add_nc_u32_e32 v19, 1, v13
.Ltmp10714:
	v_cmp_eq_u32_e32 vcc_lo, 0, v9
.Ltmp10715:
	v_mul_lo_u32 v14, v14, s9
	v_sub_nc_u32_e32 v16, v5, v16
	v_sub_nc_u32_e32 v17, v4, v17
	v_mul_lo_u32 v15, v15, s9
	v_add_nc_u32_e32 v4, 0x200, v4
	v_add_nc_u32_e32 v5, 0x200, v5
	v_cmp_le_u32_e64 s0, s8, v16
	v_cmp_le_u32_e64 s1, s8, v17
	v_sub_nc_u32_e32 v11, v11, v14
.Ltmp10716:
	s_or_b32 s7, vcc_lo, s7
.Ltmp10717:
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
.Ltmp10718:
	s_andn2_b32 exec_lo, exec_lo, s7
	s_cbranch_execnz .LBB17_110
.Ltmp10719:
	s_or_b32 exec_lo, exec_lo, s7
	v_cmp_ne_u32_e32 vcc_lo, v1, v2
	v_lshl_or_b32 v1, v2, 8, v0
	s_orn2_b32 s0, vcc_lo, exec_lo
.Ltmp10720:
.LBB17_112:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 exec_lo, exec_lo, s0
	s_cbranch_execz .LBB17_115
.Ltmp10721:
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
.Ltmp10722:
.LBB17_114:
	v_sub_nc_u32_e32 v6, 0, v1
.Ltmp10723:
	v_mul_hi_u32 v7, v5, v3
	v_ashrrev_i32_e32 v9, 31, v1
	v_max_i32_e32 v6, v6, v1
.Ltmp10724:
	v_add_nc_u32_e32 v1, 0x100, v1
.Ltmp10725:
	v_xor_b32_e32 v9, s2, v9
	v_mul_lo_u32 v7, v7, s33
	v_mul_hi_u32 v8, v6, v4
	v_sub_nc_u32_e32 v7, v5, v7
	v_mul_lo_u32 v10, v8, s1
	v_add_nc_u32_e32 v11, 1, v8
.Ltmp10726:
	v_add_nc_u32_e32 v5, 1, v5
.Ltmp10727:
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
.Ltmp10728:
	v_cndmask_b32_e32 v6, v8, v10, vcc_lo
	v_add_nc_u32_e32 v7, s4, v7
.Ltmp10729:
	v_cmp_le_i32_e32 vcc_lo, s3, v1
.Ltmp10730:
	v_xor_b32_e32 v6, v6, v9
	v_mul_lo_u32 v7, v7, s11
.Ltmp10731:
	s_or_b32 s5, vcc_lo, s5
.Ltmp10732:
	v_sub_nc_u32_e32 v6, v6, v9
.Ltmp10733:
	v_add3_u32 v6, v6, s28, v7
.Ltmp10734:
	v_lshl_or_b32 v6, v6, 8, v0
	v_ashrrev_i32_e32 v7, 31, v6
.Ltmp10735:
	v_lshlrev_b64 v[6:7], 1, v[6:7]
	v_add_co_u32 v6, s0, s34, v6
	v_add_co_ci_u32_e64 v7, null, s35, v7, s0
	global_store_short v[6:7], v2, off
.Ltmp10736:
	s_andn2_b32 exec_lo, exec_lo, s5
	s_cbranch_execnz .LBB17_114
.Ltmp10737:
.LBB17_115:
	s_endpgm
.Ltmp10738:
.Lfunc_end17:
