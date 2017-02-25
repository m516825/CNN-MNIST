import sys
import tensorflow as tf
import argparse
import csv
import struct
import numpy as np
import random
import progressbar as pb

def arg_parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('--train_dat', default='train-images.idx3-ubyte', type=str)
	parser.add_argument('--train_lab', default='train-labels.idx1-ubyte', type=str)
	parser.add_argument('--test_dat', default='test-image', type=str)
	parser.add_argument('--output', default='pred.csv', type=str)
	args = parser.parse_args()

	return args

def load_data(args):

	with open(args.train_lab, 'rb') as f:
		magic, num = struct.unpack(">II", f.read(8))
		train_lab = np.fromfile(f, dtype=np.int8)

	with open(args.train_dat, 'rb') as f:
		magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
		train_dat = np.fromfile(f, dtype=np.uint8)#.reshape(len(train_lab), rows, cols, -1)
		train_dat = train_dat.reshape(len(train_lab), rows, cols, -1)

	zip_data = list(zip(train_dat, train_lab))
	random.shuffle(zip_data)
	train_dat, train_lab = zip(*zip_data)
	train_dat, train_lab = np.array(train_dat, dtype='float32'), np.array(train_lab)
	train_dat /= 255.

	with open(args.test_dat, 'rb') as f:
		magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
		test_dat = np.fromfile(f, dtype=np.uint8)#.reshape(num, rows, cols, -1)
		test_dat = test_dat.reshape(num, rows, cols, -1)
		test_dat = np.array(test_dat, dtype='float32')
		test_dat /= 255.

	print "done loading training data, with {}".format(train_dat.shape)
	print "done loading testing data, with {}".format(test_dat.shape)

	return train_dat, train_lab, test_dat

class Data(object):
	def __init__(self, train_dat, train_lab):
		self.train_dat = train_dat
		self.train_lab = train_lab
		self.current = 0
		self.length = len(train_dat)

	def next_batch(self, size):
		if self.current == 0:
			zip_data = list(zip(self.train_dat, self.train_lab))
			random.shuffle(zip_data)
			self.train_dat, self.train_lab = zip(*zip_data)

		if self.current + size < self.length:
			batch_x, batch_y = self.train_dat[self.current:self.current+size], self.train_lab[self.current:self.current+size]
			self.current += size
		else:
			batch_x, batch_y = self.train_dat[self.current:], self.train_lab[self.current:]
			self.current = 0

		return np.array(batch_x), np.array(batch_y)

class Model(object):
	def __init__(self, train_dat, train_lab, test_dat, args):
		self.train_dat = train_dat[2000:]
		self.train_lab = train_lab[2000:]
		self.dev_dat = train_dat[:2000]
		self.dev_lab = train_lab[:2000]
		self.length = len(train_lab)
		self.test_dat = test_dat
		self.sess = tf.Session()
		self.iterations = 50
		self.batch_size = 200
		self.class_num = 10
		self.args = args

	def eval(self, model, num):
		pred = self.sess.run(model['predict'], feed_dict={model['train_x']:self.dev_dat, model['p_keep_dens']:1.})
		truth = self.dev_lab
		d_acc = np.equal(pred, truth).mean()

		pred = self.sess.run(model['predict'], feed_dict={model['train_x']:self.train_dat[:num], model['p_keep_dens']:1.})
		truth = self.train_lab[:num]
		t_acc = np.equal(pred, truth).mean()
		print "train acc: {}, dev acc: {}".format(t_acc, d_acc)

		return t_acc, d_acc

	def train(self):

		data = Data(self.train_dat, self.train_lab)
		batch_num = self.length/self.batch_size if self.length%self.batch_size == 0 else self.length/self.batch_size + 1

		model = self.add_model()

		with self.sess as sess:

			tf.initialize_all_variables().run()

			for ite in range(self.iterations):
				print "Iteration {}".format(ite)
				cost = 0.
				pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=batch_num).start()
				for i in range(batch_num):
					batch_x, batch_y = data.next_batch(self.batch_size)
					
					c, _ = self.sess.run([model['loss'], model['optimizer']], feed_dict={model['train_x']:batch_x, model['train_y']:batch_y, model['p_keep_dens']:0.75})
					
					cost += c / batch_num
					pbar.update(i+1)
				pbar.finish()

				print ">>cost: {}".format(cost)

				t_acc, d_acc = self.eval(model, 3000)
				# early stop
				if t_acc >= 0.995 and d_acc >= 0.995:
					break

			self.predict(model)

		
	def add_model(self):

		with self.sess as sess:
			train_x = tf.placeholder(tf.float32, [None, 28, 28, 1])
			train_y = tf.placeholder(tf.int32, [None])
			p_keep_dens = tf.placeholder(tf.float32)

			wc1 = tf.Variable(tf.truncated_normal([3, 3, 1, 16], stddev=0.1))
			bc1 = tf.Variable(tf.constant(0.1, shape=[16]))
			wc2 = tf.Variable(tf.truncated_normal([3, 3, 16, 16], stddev=0.1))	
			bc2 = tf.Variable(tf.constant(0.1, shape=[16]))

			wd1 = tf.Variable(tf.truncated_normal([5*5*16, 256], stddev=0.1))	# Dense 1536 * 512
			bd1 = tf.Variable(tf.constant(0.1, shape=[256]))
			wd2 = tf.Variable(tf.truncated_normal([256, 128], stddev=0.1))	# Dense 1536 * 512
			bd2 = tf.Variable(tf.constant(0.1, shape=[128]))
			wd_out = tf.Variable(tf.truncated_normal([128, self.class_num], stddev=0.1))	# output layer 512 * class
			bd_out = tf.Variable(tf.constant(0.1, shape=[self.class_num]))

			c_layer1 = tf.nn.conv2d(train_x, wc1, strides=[1, 1, 1, 1], padding="VALID") # (28-3+1)/1 = 24
			c_layer1 = tf.nn.relu(tf.nn.bias_add(c_layer1, bc1))
			c_layer1 = tf.nn.max_pool(c_layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID') #23/2 = 12
			# c_layer1 = tf.nn.dropout(c_layer1, p_keep_dens)

			c_layer2 = tf.nn.conv2d(c_layer1, wc2, strides=[1, 1, 1, 1], padding="VALID") # (12-3+1)/1 = 10
			c_layer2 = tf.nn.relu(tf.nn.bias_add(c_layer2, bc2))
			c_layer2 = tf.nn.max_pool(c_layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID') # 10/2 = 5
			# c_layer2 = tf.nn.dropout(c_layer2, p_keep_dens)

			c_layer_out = tf.reshape(c_layer2, [-1, wd1.get_shape().as_list()[0]])
			c_layer_out = tf.nn.dropout(c_layer_out, p_keep_dens)

			ld1 = tf.add(tf.matmul(c_layer_out, wd1), bd1)
			ld1 = tf.nn.relu(ld1)
			ld1 = tf.nn.dropout(ld1, p_keep_dens)

			ld2 = tf.add(tf.matmul(ld1, wd2), bd2)
			ld2 = tf.nn.relu(ld2)
			ld2 = tf.nn.dropout(ld2, p_keep_dens)

			out = tf.add(tf.matmul(ld2, wd_out), bd_out)

			loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(out, train_y))
			optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

			predict = tf.argmax(out, 1)

		return {
				'train_x':train_x,
				'train_y':train_y,
				'optimizer':optimizer,
				'loss':loss,
				'predict':predict,
				'p_keep_dens':p_keep_dens
				}

	def predict(self, model):

		pred = self.sess.run(model['predict'], feed_dict={model['train_x']:self.test_dat, model['p_keep_dens']:1.})
		f = open(self.args.output, 'w')
		f.write('id,label\n')
		for i, p in enumerate(pred):
			f.write(str(i)+','+str(int(p))+'\n')
	
def main():

	args = arg_parse()

	train_dat, train_lab, test_dat = load_data(args)

	model = Model(train_dat, train_lab, test_dat, args)

	model.train()


if __name__ == "__main__":
	main()