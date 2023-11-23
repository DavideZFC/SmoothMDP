def plot_label(label, color, filename = 'TeX/template_plot.txt'):
    with open(filename, 'r') as file:
        # read in the contents of the file
        contents = file.read()
    file.close()

    # replace all occurrences of 'H' with 'my_word'
    contents = contents.replace('H', label)

    # replace all occurrences of 'K' with 'my_other_word'
    contents = contents.replace('K', color)

    return contents

def add_file(filename='TeX/reference_tex.txt'):    
    with open(filename, 'r') as file:
        content = file.read()
    file.close()
    return content




with open('Tex/new_file.txt', 'w') as new_file:
    new_file.write(add_file())
    new_file.write(plot_label('UCB','blue'))
    new_file.write(plot_label('UCB','blue', filename = 'TeX/template_fill.txt'))
    new_file.write(add_file('TeX/refrence_end.txt'))

new_file.close()