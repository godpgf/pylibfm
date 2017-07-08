all:
	cd libfm; make all
	cd pyfm; python setup.py install

clean:
	cd libfm; make clean



