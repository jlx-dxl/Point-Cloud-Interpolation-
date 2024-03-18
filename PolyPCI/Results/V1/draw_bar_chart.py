import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file


# Extract the date column from the DataFrame
data = pd.read_csv('field_3.csv')
field_3_degree_1 = np.mean(np.asarray(data['field=3,degree=1']).reshape(-1, 31), axis=0)
print(field_3_degree_1.shape)
field_3_degree_2 = np.mean(np.asarray(data['field=3,degree=2']).reshape(-1, 31), axis=0)
print(field_3_degree_2.shape)
field_3_degree_3 = np.mean(np.asarray(data['field=3,degree=3']).reshape(-1, 31), axis=0)
print(field_3_degree_3.shape)
field_3_degree_4 = np.mean(np.asarray(data['field=3,degree=4']).reshape(-1, 31), axis=0)
print(field_3_degree_4.shape)
field_3_degree_5 = np.mean(np.asarray(data['field=3,degree=5']).reshape(-1, 31), axis=0)
print(field_3_degree_5.shape)
field_3_degree_6 = np.mean(np.asarray(data['field=3,degree=6']).reshape(-1, 31), axis=0)
print(field_3_degree_6.shape)

# index = ['-1.0', '-0.8', '-0.6', '-0.4', '-0.2', '0', '0.2', '0.4', '0.6', '0.8', '1.0']
# Generate the array
array = np.arange(-3, 3.2, 0.2)
print(array.shape)


width=0.03
ax1 = plt.subplot(211)
plt.bar(array-2.5*width, field_3_degree_1,width=width,label='field=3,degree=1')
plt.bar(array-1.5*width, field_3_degree_2,width=width,label='field=3,degree=2')
plt.bar(array-0.5*width, field_3_degree_3,width=width,label='field=3,degree=3')
plt.bar(array+0.5*width, field_3_degree_4,width=width,label='field=3,degree=4')
plt.bar(array+1.5*width, field_3_degree_5,width=width,label='field=3,degree=5')
plt.bar(array+2.5*width, field_3_degree_6,width=width,label='field=3,degree=6')
plt.legend(loc='upper center')

ax2 = plt.subplot(212)
plt.plot(array, field_3_degree_1,label='field=3,degree=1',marker="o")
plt.plot(array, field_3_degree_2,label='field=3,degree=2',marker="*")
plt.plot(array, field_3_degree_3,label='field=3,degree=3',marker="s")
plt.plot(array, field_3_degree_4,label='field=3,degree=4',marker="v")
plt.plot(array, field_3_degree_5,label='field=3,degree=5',marker="x")
plt.plot(array, field_3_degree_6,label='field=3,degree=6',marker="d")
plt.legend(loc='upper center')

# # Count the occurrences of each date
# date_counts = dates.value_counts().sort_index()
#
# # Plotting the bar chart
# plt.bar(date_counts.index, date_counts.values)
#
# # Formatting the x-axis labels as dates
# plt.xticks(rotation=45)
#
# # Adding labels and title
# plt.xlabel('Date')
# plt.ylabel('Count')
# plt.title('Date Frequency')

#
# # Display the chart
plt.show()
