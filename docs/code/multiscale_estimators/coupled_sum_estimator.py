from structure_factor.multiscale_estimators import coupled_sum_estimator

y_list= [3, 2.5, 1.2433, 0.1]
proba_list = [0.5, 0.4, 0.333, 0.21232]
z = coupled_sum_estimator(y_list, proba_list)
z