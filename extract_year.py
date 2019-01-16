import re

d = re.findall('(\d{4})', 'Arte poetice ale secolului XX. Ipostaze românești și străine, București, 1976')
print(d[-1])
print('---')

d = re.findall('(\d{4})', 'Scriitori maghiari din România. 1920–1980, București, 1981')
print(d[-1])
print('---')

d = re.findall('(\d{4})', 'De la Homer la Joyce, București, 2007')
print(d[-1])
print('---')