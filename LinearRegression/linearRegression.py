from numpy import *

from matplotlib import pyplot as plt

#squared error function or cost function
def computer_error_for_given_points(b, m, points):
        totalError = 0
        for i in range(0, len(points)):
                x = points[i,0]
                y = points[i,1]
                totalError += (y - (m*x + b))**2
        return totalError / float(len(points))
                
        

def step_gradient(b_current, m_current, points, learningRate):
	#gradient descent
        b_gradient = 0
        m_gradient = 0
        N = float(len(points))
        for i in range(0,len(points)):
                x = points[i,0]
                y = points[i,1]
                b_gradient += -(2/N) * (y- ((m_current*x) + b_current) )
                m_gradient += -(2/N) * x * (y- ((m_current*x) + b_current) )
        new_b = b_current - (learningRate* b_gradient)
        new_m = m_current - (learningRate* m_gradient)
        #simultaneous Update should be maintained i.e both derivative should be taken
        #at once then we will update b and m together. * coursera Gradient Descent Slide 1
        return [new_b,new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
	b = starting_b
	m = starting_m
	
	for i in range(num_iterations):
		b,m = step_gradient(b, m, array(points), learning_rate)
	return ([b,m])	

def run():
        #collection of test score and amnt of hour study
        # x_values = amt of hour studied
        # y_values = score
	points = genfromtxt('data.csv',delimiter=",")
	#hyperparameters (Alpha)
	
	learning_rate = 0.0001

	#y = mx + b
	initial_b = 0
	initial_m = 0
	num_iterations = 1000
	#optimal value of b and m
	[b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
	print(b)
	print(m)
	x = [points[i,0] for i in range(len(points))]
	y = [points[i,1] for i in range(len(points))]
	plt.scatter(x,y,color='green')
	plt.plot([min(x) , max(x)] , [m*min(x)+b , m*max(x) + b] , 'r')
	plt.xlabel('# hr studied')
	plt.ylabel('marks')
	plt.show()
	

	


if __name__ ==  "__main__":
	run()



