import os

import matplotlib.pyplot as plt


class PlotDiar:
    """
    A viewer of segmentation
    """

    def __init__(self,
                 true_map=None,
                 map=None,
                 wav=None,
                 gui=False,
                 pick=False,
                 size=(24, 6)
                 ):
        # Default params
        self.true_map = true_map
        self.map = map
        self.wav = wav
        self.gui = gui
        self.pick = pick
        self.size = size

        # Setup plot
        self.plt = plt
        self.fig = plt.figure(figsize=size, facecolor='white', tight_layout=True)

        # Setup axes for true and generated segments
        self.true_ax = self.fig.add_subplot(2, 1, 1)
        self.ax = self.fig.add_subplot(2, 1, 2)

        # Setup timelines
        self.true_timeline = self.true_ax.plot([0, 0], [0, 0], color='r')[-1]
        self.timeline = self.ax.plot([0, 0], [0, 0], color='r')[-1]

        # Setup other params
        self.segments_colors = ['#ffd740', '#9c27b0', '#00bcd4', '#00e676', '#ffff00', '#e91e63']

        self.true_timestamps = []
        self.timestamps = []
        self.true_timestamp_index = 0
        self.timestamp_index = 0

        self.rect_picked = None
        self.rect_color = '#ffd740'

        self.height = 5
        self.max_x = 0
        self.max_y = 0
        self.end_play = 0

    def _draw_info(self, ax, xlabel, ylabel, title):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def _draw_segments(self, ax, map, timestamps):
        labels_position = []
        labels = []
        y = 0
        self.max_y = 0

        for i, cluster in enumerate(sorted(map.keys())):
            labels.append(cluster)
            labels_position.append(y + self.height // 2)

            for row in map[cluster]:
                x = row['start'] / 1000

                timestamps.append(x)
                timestamps.append(row['stop'] / 1000)

                w = row['stop'] / 1000 - row['start'] / 1000

                self.max_x = max(self.max_x, row['stop'] / 1000)

                c = self.segments_colors[i % len(self.segments_colors)]
                rect = plt.Rectangle((x, y), w, self.height, color=c, picker=self.pick)
                ax.add_patch(rect)

            y += self.height

        if self.gui:
            ax.set_xlim([0, min(600, self.max_x)])
        else:
            ax.set_xlim([0, self.max_x])

        ax.set_ylim([0, y])
        ax.set_yticks(labels_position)
        ax.set_yticklabels(labels)
        self.max_y = y
        self.end_play = self.max_x

        for cluster in map:
            ax.plot([0, self.max_x], [y, y], linestyle=':', color='#AAAAAA')
            y -= self.height

        timestamps = list(set(self.timestamps)).sort()

        return timestamps

    def draw_true_map(self):
        self._draw_info(self.true_ax, 'True segments', 'True speakers', f'True diarization map for {self.wav}')
        self._draw_segments(self.true_ax, self.true_map, self.true_timestamps)

    def draw_map(self):
        self._draw_segments(self.ax, self.map, self.timestamps)
        self._draw_info(self.ax, 'Detected segments', 'Detected speakers', f'Detected diarization map for {self.wav}')

    def save(self, dir, plot_name='plot'):
        self.fig.savefig(os.path.join(dir, plot_name))

    def show(self):
        self.plt.tight_layout()
        self.plt.show()

    @classmethod
    def _hms(cls, seconds):
        """
        Conversion of seconds into hours, minutes and seconds
        :param seconds:
        :return: int, int, float
        """
        h = int(seconds) // 3600
        seconds %= 3600
        m = int(seconds) // 60
        seconds %= 60

        return f'{h:d}:{m:d}:{seconds:.2f}'

    @classmethod
    def _colors_are_equal(cls, c1, c2):
        """
        Compare two colors
        """
        for i in range(4):
            if c1[i] != c2[i]:
                return False
        return True
