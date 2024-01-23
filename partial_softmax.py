import numpy as np
from scipy.special import softmax

# np.random.seed(42)
L, D = 2, 10
x = np.random.randn(2,10) # sequence (L), dim (D)
print("input : ", x)

b = 5 # number of blocks
assert x.shape[-1]%b == 0, f"number of blocks {b} is not an \
equal division of the the input array size {x.shape[-1]}"
B = x.shape[-1]//b # size

original_softmax = softmax(x)
print("original softmax : ", original_softmax)


def get_partial_softmax(_input:np.ndarray):
	m_input = np.max(_input)
	f_input = np.e**(_input-m_input)
	l_input = np.sum(f_input)
	return m_input, (f_input, l_input)

partial = {}
M_x = np.zeros(b)

for i in range(B, len(x)+1, B):
	m_x, partial[i//B - 1] = get_partial_softmax(x[i-B:i])
	M_x[i//B - 1] = m_x

F_x = np.empty([len(partial), *partial[0][0].shape])
L_x = np.empty_like(F_x)

for i, val in partial.items():

	f_x, l_x = val

	F_x[i] = (np.e**(M_x[i] - max(M_x)))*(f_x)
	L_x[i] = (np.e**(M_x[i] - max(M_x)))*(l_x)


L_x = np.sum(L_x, axis=0)
partial_softmax = (F_x/L_x).reshape(L,D)
print("partial softmax : ", partial_softmax)
print("original_softmax == partial_softmax =>",\
 np.allclose(original_softmax, partial_softmax))

###########################################################################
