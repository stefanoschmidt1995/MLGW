#!/bin/bash

VERSION="2"
FOLDER="../mlgw_v$VERSION"

echo "version = $VERSION"

cp "$FOLDER/EM_MoE.py" ./mlgw
cp "$FOLDER/fit_model.py" ./mlgw
cp "$FOLDER/GW_helper.py" ./mlgw
cp "$FOLDER/GW_generator.py" ./mlgw
cp "$FOLDER/ML_routines.py" ./mlgw
echo copied package files: EM_MoE.py fit_model.py GW_helper.py GW_generator.py ML_routines.py

if [ "$VERSION" = "1" ]; then 
	rm -r -f ./mlgw/TD_models/*
		#model_3 (SEOBNRv4)
	cp -r ../mlgw_v1/TD_model_SEOBNRv4 ./mlgw/TD_models/model_3
		#model_2 (thesis model)
	cp -r ../mlgw_v1/TD_model_thesis ./mlgw/TD_models/model_2
		#model_1 (TEOBResumS default)
	cp -r ../mlgw_v1/TD_model_TEOBResumS_long ./mlgw/TD_models/model_1
		#model_0 (TEOBResumS default)
	cp -r ../mlgw_v1/TD_model_TEOBResumS ./mlgw/TD_models/model_0
fi
if [ "$VERSION" = "2" ]; then
	rm -rf ./mlgw/TD_models 
	cp -r ../mlgw_v2/TD_models ./mlgw
fi
echo copied models









