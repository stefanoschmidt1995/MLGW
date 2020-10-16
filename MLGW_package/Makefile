
update_package:
	cp code/routines/EM_MoE.py code/MLGW_package/mlgw
	cp code/routines/fit_model.py code/MLGW_package/mlgw
	cp code/routines/GW_helper.py code/MLGW_package/mlgw
	cp code/routines/GW_generator.py code/MLGW_package/mlgw
	cp code/routines/ML_routines.py code/MLGW_package/mlgw
	rm -r -f code/MLGW_package/mlgw/TD_models/*
	#model_3 (SEOBNRv4)
	cp -r code/definitive_code/TD_model_SEOBNRv4 code/MLGW_package/mlgw/TD_models/model_3
	#model_2 (thesis model)
	cp -r code/definitive_code/TD_model_thesis code/MLGW_package/mlgw/TD_models/model_2
	#model_1 (TEOBResumS default)
	cp -r code/definitive_code/TD_model_TEOBResumS_long code/MLGW_package/mlgw/TD_models/model_1
	#model_0 (TEOBResumS default)
	cp -r code/definitive_code/TD_model_TEOBResumS code/MLGW_package/mlgw/TD_models/model_0

