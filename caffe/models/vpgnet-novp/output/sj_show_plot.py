'''
Seokju Lee, 16.08.05
Description: parse output.log and plot training loss and validation accuracy
Tune the nth-scan parameter 'nth'
'''
import numpy as np
import pdb
import matplotlib.pyplot as plt


def close_event():
	plt.close()


while 1: 
	logPath = './output.log'
	logFile = open(logPath,'r')
	print '%s' %(logPath)

	out_0 = []
	out_1 = []
	out_2 = []
	out_3 = []

	acc_0 = []
	acc_1 = []
	acc_2 = []

	nth = 14	# Tune it!

	for line in logFile.readlines():
		if ('Train net output #0:' in line):
			data = line.split(' ')
			out_0.append(data[nth])

		if ('Train net output #1:' in line):
			data = line.split(' ')
			out_1.append(data[nth])

		if ('Train net output #2:' in line):
			data = line.split(' ')
			out_2.append(data[nth])

		if ('Train net output #3:' in line):
			data = line.split(' ')
			out_3.append(data[nth])

		if ('Test net output #1:' in line):
			data = line.split(' ')
			acc_0.append(data[nth])

		if ('Test net output #3:' in line):
			data = line.split(' ')
			acc_1.append(data[nth])

		if ('Test net output #5:' in line):
			data = line.split(' ')
			acc_2.append(data[nth])

	arr_x = range(0,1000,10)
	# pdb.set_trace()
	arr_out_0 = np.float32( out_0 )
	arr_out_1 = np.float32( out_1 )
	arr_out_2 = np.float32( out_2 )
	arr_out_3 = np.float32( out_3 )

	arr_acc_0 = np.float32( acc_0 )
	arr_acc_1 = np.float32( acc_1 )
	arr_acc_2 = np.float32( acc_2 )

	# pdb.set_trace()

	fig = plt.figure(figsize=(8, 16))

	fig.add_subplot(7,1,1)
	plt.plot(arr_out_0)
	plt.grid()
	plt.xlabel("iterations")
	plt.ylabel("bb_loss")
	frame = plt.gca()
	# frame.set_ylim([0, 5])

	fig.add_subplot(7,1,2)
	plt.plot(arr_out_1)
	plt.grid()
	plt.xlabel("iterations")
	plt.ylabel("px_loss")
	frame = plt.gca()
	# frame.set_ylim([0, 0.5])

	fig.add_subplot(7,1,3)
	plt.plot(arr_out_2)
	plt.grid()
	plt.xlabel("iterations")
	plt.ylabel("type_loss")
	frame = plt.gca()
	# frame.set_ylim([0, 1.0])

	fig.add_subplot(7,1,4)
	plt.plot(arr_out_3)
	plt.grid()
	plt.xlabel("iterations")
	plt.ylabel("vp_loss")
	frame = plt.gca()
	# frame.set_ylim([0.00, 0.015])

	fig.add_subplot(7,1,5)
	plt.plot(arr_acc_0)
	plt.grid()
	plt.xlabel("epochs")
	plt.ylabel("px_acc")
	frame = plt.gca()
	# frame.set_ylim([0.9, 1])

	fig.add_subplot(7,1,6)
	plt.plot(arr_acc_1)
	plt.grid()
	plt.xlabel("epochs")
	plt.ylabel("type_acc")
	frame = plt.gca()
	# frame.set_ylim([0.9, 1])

	fig.add_subplot(7,1,7)
	plt.plot(arr_acc_2)
	plt.grid()
	plt.xlabel("epochs")
	plt.ylabel("vp_acc")
	frame = plt.gca()
	# frame.set_ylim([0.995, 1])


	fig.tight_layout()
	fig.savefig('sj_show_plot.png')
	timer = fig.canvas.new_timer(interval = 1000 * 60 * 3) #interval=3000: creating a timer object and setting an interval of 3000 milliseconds
	timer.add_callback(close_event)
	timer.start()

	plt.show()

	# pdb.set_trace()