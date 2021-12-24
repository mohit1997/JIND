import os
import numpy as np

def main():
	path = "datasets/pancreas_raw_01"
	# path = "datasets/human_blood_01"
	print(path)
	
	jind = {"raw": [], "rej": [], "eff": [], "mf1": [], "medf1": [], "wf1": []}
	jindplus = {"raw": [], "rej": [], "eff": [], "mf1": [], "medf1": [], "wf1": []}

	for i in range(5):
		filename = os.path.join(path, f"JIND_rawtop_{i}", "test.log")
		with open(filename) as f:
			content = f.readlines()
		
		lines = [x.strip() for x in content]
		
		jindresults = lines[1].split(" ")

		jind["raw"].append(np.float(jindresults[4]))
		jind["eff"].append(np.float(jindresults[6]))
		jind["rej"].append(np.float(jindresults[8]))
		jind["mf1"].append(np.float(jindresults[10]))
		jind["medf1"].append(np.float(jindresults[12]))
		jind["wf1"].append(np.float(jindresults[14]))

		jindplusresults = lines[2].split(" ")

		jindplus["raw"].append(np.float(jindplusresults[4]))
		jindplus["eff"].append(np.float(jindplusresults[6]))
		jindplus["rej"].append(np.float(jindplusresults[8]))
		jindplus["mf1"].append(np.float(jindplusresults[10]))
		jindplus["medf1"].append(np.float(jindplusresults[12]))
		jindplus["wf1"].append(np.float(jindplusresults[14]))


	for method in [jind, jindplus]:
		if method == jind:
			print("JIND")
		elif method == jindplus:
			print("JIND+")
		for key in method.keys():
			print(f"{key}: Mean {np.mean(method[key]):.3f} Std {np.std(method[key]):.3f} Median {np.median(method[key]):.3f}")

if __name__ == "__main__":
	main()
