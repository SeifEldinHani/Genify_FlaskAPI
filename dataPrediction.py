import pickle
import xgboost as xgb
import numpy as np

class dataPredicition:
    def __init__(self , model_name) :
        self.model = pickle.load(open(model_name,'rb'))
        self.target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
        self.target_cols = np.array(self.target_cols[2:])
    
    def predict(self , data):
        data = np.array(data)
        shape = data.shape[0]
        final_data = data.reshape(1,shape)
        xgtest  = xgb.DMatrix(final_data)
        preds = self.model.predict(xgtest)
        preds = np.argsort(preds, axis=1)
        preds = np.fliplr(preds)[:,:7]
        final_preds = [" ".join(list(self.target_cols[pred])) for pred in preds]
        return final_preds[0]
        


