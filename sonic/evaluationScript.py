
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import sonic_multi_input
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import math


def main(rewardList, experiments, timesteps,epsilon,decay):
	scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
	creds = ServiceAccountCredentials.from_json_keyfile_name("Creds.json", scope)
	client = gspread.authorize(creds)
	sheet = client.open("SonicTable").sheet1  # Open the spreadhseet
	data = sheet.get_all_records()
	print (data)
	for e in range (1, experiments, 1):
		#if e%experiments == 0:
			ci = 0.95 # 95% confidence interval
			means = np.mean(rewardList, axis=0)
			stds = np.std(rewardList, axis=0)
			n = means.size
			# compute upper/lower confidence bounds
			test_stat = st.t.ppf((ci + 1) / 2, e)
			lower_bound = means - test_stat * stds / np.sqrt(e)
			upper_bound = means + test_stat * stds / np.sqrt(e)

			print ('Avg. Reward per step in experiment %d: %.4f' % (e, sum(means) / timesteps))
			# clear plot frame
			plt.clf()
			# plot average reward
			plt.plot(means, color='blue', label="epsilon=%.2f" % epsilon)
			# plot upper/lower confidence bound
			x = np.arange(0, timesteps, 1)
			plt.fill_between(x=x, y1=lower_bound, y2=upper_bound, color='blue', alpha=0.2, label="CI %.2f" % ci)
			plt.grid()
			plt.ylim(0, 2) # limit y axis
			#plt.title('Avg. Reward per step (UCB with c=%.2f, decay=%.1e) in experiment %d: %.4f' % (c ,decay, e, sum(means) / timesteps))
			plt.ylabel("Reward per step")
			plt.xlabel("Play")
			plt.legend()
			plt.show()
	plt.ioff()
	plt.show()

if __name__ == "__main__":
    main()