
#Train polar optimised model
$python3 FinalPolar-train.py

# test response time
$python FinalPolar-singletestEval.py

#train and analysis 
$python3 FinalTrainisolatedrun.py

(
# Use this command to expand dataset into R,G,B 
$python3 RGBgen.py
)


