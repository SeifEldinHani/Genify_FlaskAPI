from dataPrediction import dataPredicition
from flask import Flask
from flask_restx import Resource, Api , fields
from dataProcessing import dataProcessing
from dataPrediction import dataPredicition

app = Flask(__name__)


#Classes instants for processing class as well as predection class  
processing = dataProcessing()
model = dataPredicition("finalized_model.sav")

api = Api(app , title = "Banking recommendation API" , description='Banking recommendation API is an API built to predict the best products for customers based on customers\' data')

#Namespace for predections 
Predections_namespace = api.namespace('Predictor')

#Request Variables 
Request_Fields = api.model('Request-Fields', {
    'sexo': fields.String(description="Customer's sex", enum  = ["V" , "H"] , required=True),
    'age': fields.Integer(description="Customer's Age",required=True),
    'segmento': fields.String(description = "segmentation: 01 - VIP, 02 - Individuals 03 - college graduated" , 
    enum= ['01 - TOP' , '02 - PARTICULARES' , '03 - UNIVERSITARIO'] , required = True),
    'antiguedad': fields.Integer(description = "Customer seniority (in months)" , required=True),
    'tiprel_1mes' : fields.String(description = "Customer relation type at the beginning of the month, A (active), I (inactive), P (former customer),R (Potential)",
     enum = ["A","I","P","R"] , required = True),
    "renta" : fields.Float(description = "Gross income of the household" , required = True), 
    "pais_residencia" : fields.String(description = "Customer's Country residence" , required = True), 
    "ind_actividad_cliente" : fields.String(description = "Activity index ('1', active customer; '0', inactive customer)" , enum = ["1" , "0"] , required = True),
    "nomprav" : fields.String(description = "Province name" , required = True)

})

#Response Variables
Response_Fields = api.model('Response-Fields' , {
    'Predections' : fields.String(description = "Model's predictions of best products")
})

#Prediction API 
@Predections_namespace.route('/Predict')
@Predections_namespace.doc()
class Predector(Resource):
    @Predections_namespace.response(200, 'Success', Response_Fields)
    @Predections_namespace.response(400, 'Validation Error')
    @Predections_namespace.expect(Request_Fields , validate = True)    
    def post(self):
        predictions = model.predict(processing.processData(api.payload))
        return {'Predictions' : predictions}, 200


if __name__ == '__main__':
    app.run(debug=True)