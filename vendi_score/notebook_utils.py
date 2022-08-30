import copy

from IPython.display import display, Markdown, Latex


def markdown_table(headers, rows, bold_left=True, do_display=True):
    lines = [
        headers,
        ([":---"] + [":---:" for _ in range(len(headers) - 2)] + ["---:"]),
    ] + rows
    if bold_left:
        lines = [copy.copy(l) for l in lines]
        for l in lines[2:]:
            l[0] = f"**{l[0]}**"
    out = "\n".join(["|".join(list(map(str, l))) for l in lines])
    if do_display:
        display(Markdown(out))
    else:
        return out
