import pandas as pd
import ManhattanDetector as manhattanDetector
import visual as vs

df = pd.read_excel('DSL-StrongPasswordData.xls')
users = df['subject'].unique()
sample_size = int(input('Enter the size of the sample: '))
threshold = float(input('Enter a threshold value: '))
threshold_selected = [0,1,2,3,8]

fpr, ipr, fpr_selected, ipr_selected, ipr_at_zero_fpr, fpr_at_zero_ipr, impostor_score, user_score, eet, eer_selected =\
    manhattanDetector.ManhattanDetector(df, sample_size, 0, users, threshold, threshold_selected)

print()
print("---------------------------------------------------------")
print("False positive rate (false reject rate) for threshold of"+" {:.2f} ".format(threshold)+" (N="+"{}".format(sample_size)+") is" +
      " {:.3} ".format(fpr))
print("Impostor pass rate (false accept rate) for threshold of"+" {:.2f} ".format(threshold)+" (N="+"{}".format(sample_size)+") is" +
      " {:.3} ".format(ipr))
print(" ")
print("--------------------------------------------------------")
print(" ")
print("At selected thresholds " + format(threshold_selected))

for i in range(len(threshold_selected)):
    print("False positive rate (false reject rate) for threshold of" + " {:.2f} ".format(threshold_selected[i]) +
          "(N="+"{}".format(sample_size) + ") is" + " {:.3} ".format(fpr_selected[i]))
    print("Impostor pass rate (false accept rate) for threshold of" + " {:.2f} ".format(threshold_selected[i]) +
          "(N="+"{}".format(sample_size) + ") is" + " {:.3} ".format(ipr_selected[i]))
    print(" ")

print()
print("---------------------------------------------------------")

print("Impostor pass rate (false accept rate) at zero false reject rate is (N="+"{}".format(sample_size) + ") {:.5f}".format(ipr_at_zero_fpr)+" at threshold (max score of genuine) "+ "{:.3}".format(user_score['Score'].max()))
print("At the same threshold, false reject rate is (verification): " + "{}".format(manhattanDetector.false_postive_rate(user_score,user_score.max())))
print()
print("False positive rate (false reject rate) at zero impostor pass rate is (N="+"{}".format(sample_size) + "){:.5f}".format(fpr_at_zero_ipr)+" at threshold (minimum score impostor) "+ "{:.3f}".format(impostor_score['Score'].min()*0.99))
print("At the same threshold, impostor pass rate is (verification): " + "{}".format(manhattanDetector.impostor_pass_rate(impostor_score,impostor_score['Score'].min()*0.99)))

vs.roc_curve_population(user_score,impostor_score, sample_size)

vs.det_curve_population(threshold_selected,fpr_selected, ipr_selected,user_score,impostor_score, sample_size,2)

print()
print("---------------------------------------------------------")

print("Equal error rate for selected threshold (N=" + "{}".format(sample_size) + ") is " + "{:.3f}".format(eet))
print("Threshold at which equal error rate occurs (N=" + " {}".format(sample_size) + ") is {:.3f}".format(eer_selected))

