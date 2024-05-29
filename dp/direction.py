import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_piecewise_with_directions(segments):
    arrow_y = min(seg[2] for seg in segments) - 1
    for i, (x_start, x_end, y) in enumerate(segments):
        plt.plot([x_start, x_end], [y, y], '-k')

        if 0 < i < len(segments) - 1:
            cur_y = segments[i][2]
            prev_y = segments[i - 1][2]
            next_y = segments[i + 1][2]

            # Determine arrow color based on the direction and relative difference
            if cur_y > prev_y or cur_y > next_y:
                if next_y > prev_y:
                    delta = cur_y - next_y
                    color = mcolors.to_rgba('red', alpha=min(1, abs(delta) / (max(cur_y, next_y) - arrow_y)))
                    plt.annotate('', xy=(x_start, arrow_y), xytext=(x_end, arrow_y),
                                 arrowprops=dict(arrowstyle='->', color=color))
                else:
                    delta = cur_y - prev_y
                    color = mcolors.to_rgba('blue', alpha=min(1, abs(delta) / (max(cur_y, prev_y) - arrow_y)))
                    plt.annotate('', xy=(x_end, arrow_y), xytext=(x_start, arrow_y),
                                 arrowprops=dict(arrowstyle='->', color=color))

        if i == 0 and segments[i][2] > segments[i + 1][2]:
            delta = segments[i][2] - segments[i + 1][2]
            color = mcolors.to_rgba('blue',
                                    alpha=min(1, abs(delta) / (max(segments[i][2], segments[i + 1][2]) - arrow_y)))
            plt.annotate('', xy=(x_end, arrow_y), xytext=(x_start, arrow_y),
                         arrowprops=dict(arrowstyle='->', color=color))

        if i == len(segments) - 1 and segments[i - 1][2] < segments[i][2]:
            delta = segments[i][2] - segments[i - 1][2]
            color = mcolors.to_rgba('red',
                                    alpha=min(1, abs(delta) / (max(segments[i][2], segments[i - 1][2]) - arrow_y)))
            plt.annotate('', xy=(x_start, arrow_y), xytext=(x_end, arrow_y),
                         arrowprops=dict(arrowstyle='->', color=color))

    plt.xlim(min(seg[0] for seg in segments) - 1, max(seg[1] for seg in segments) + 1)
    plt.ylim(min(seg[2] for seg in segments) - 2, max(seg[2] for seg in segments) + 2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


segments_example = [
    (0, 3, 1),
    (3, 4, 3),
    (4, 6, 2),
    (6, 9, 1),
    (9, 12, 4)
]

plot_piecewise_with_directions(segments_example)
