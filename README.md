### Please cite Milo as:
Milo, Revision 1.0.3, M. S. Teynor, N. Wohlgemuth, L. Carlson, J. Huang, S. L. Pugh, B. O. Grant, R. S. Hamilton, R. Carlsen, and D. H. Ess, Brigham Young University, Provo UT, 2021.

### Requirements
#### Python:
Milo has only been tested against Python 3.8. It is expected to work with Python 3.6+, but will not work with Python 3.5 or older.

#### Gaussian:
Milo can interface with Gaussian 16, Gaussian 09, and ORCA to perform force calculation. Milo expects Gaussian or ORCA to be loaded as a module. The executable for the corresponding QM software package is passed to Milo on the command line. 

#### AaronTools:
Use the version on <a href="https://github.com/QChASM/AaronTools.py/wiki/Installation#download-from-github">GitHub</a>.

#### Operating System:
It should work

### Installation Guide
1. Download <a href="https://github.com/ajs99778/milo/raw/main/dist/Milo-1.0.4-py3-none-any.whl">Milo-1.0.4-py3-none-any.whl</a>
2. On the terminal run `python -m pip install Milo-1.0.4-py3-none-any.whl`

You are now ready to run your first Milo job.  

### Using Milo
To run a Milo job with Gaussian, the input file should have:
```
$job
	program	gaussian
	...
$end
```
and to run Milo:
```sh
	module load python/3.8
	module load gaussian
	python -m milo job.in -e g16
```


To run a Milo job with ORCA, the input file should have:
```
$job
	program	orca
	...
$end
```
and to run Milo:
```sh
	module load python/3.8
	module load orca
	python -m milo job.in -e `which orca`
```

Level of theory is specified using the `$theory` block in the input file. Example:
```
$theory
	method			B3LYP
	basis			def2-SVP
	empirical_dispersion	D3Zero
	grid			SuperFineGrid
	processors		16 # this will be moved back to $job eventually
$end
```
