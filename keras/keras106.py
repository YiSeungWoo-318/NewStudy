weight=0.5
input=0.5
goal_prediction=0.8

Ir=0.001

for iteration in range(1101):
    prediction=input*weight
    error=(prediction-goal_prediction)**2

print("Error:" + str(error)+ "\tPrediction:" + str(prediction))
up_prediction=input * (weight+Ir)
up_error=(goal_prediction-up_prediction)**2

down_prediction=input*(weight-Ir)
down_error = (goal_prediction-down_prediction) ** 2

if(down_error < up_error) :
    weight =weight - Ir
if(down_error > up_error) :
    weight=weight+Ir

