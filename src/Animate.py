import animatplot as amp
import matplotlib.pyplot as plt
import numpy as np


def generateAnimat(records: [[float]], fps: int = 1, vmin=0, vmax=255, generate_gif: bool = False, filename: str = 'animat'):

	fig, ax = plt.subplots()

	def animate(i):

		ax.set_xlabel('X')
		ax.set_ylabel('Y')

		# Generate the pix map
		ax.imshow(records[i], interpolation='none', cmap='jet', vmin=vmin, vmax=vmax)

	blocks = amp.blocks.Nuke(animate, length=len(records), ax=ax) # Required call to build our animation
	timeline = np.arange(len(records))
	anim = amp.Animation([blocks], amp.Timeline(timeline, fps=fps))  # Builds the Animation
	anim.controls()  # Gives us a pause and start button
	
	if generate_gif:  # Write animation to file if generate_gif is True
		anim.save_gif(filename)

	return anim, fig, ax


if __name__ == '__main__':

	infile = "./model_5000.dat"
	outfile = "./resources/house_movement_5000"

	height = 200
	width = 200

	records = []


	with open(infile, 'r') as f:
		for line in [l.rstrip() for l in f.readlines()]:

			splitData = [int(data) for data in line.split(' ')]
			record = []
			counter = 0
			for h in range(height):
				row = []
				for w in range(width):
					row.append(splitData[counter])
					counter += 1

				record.append(row)

			records.append(record)

	generateAnimat(records, vmin=0, vmax=1, fps=10, filename=outfile, generate_gif=True)

