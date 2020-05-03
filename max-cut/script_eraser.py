import subprocess

if __name__ == '__main__':
    script_id = input('Enter script_id\n')
    subprocess.getoutput('rm t_{}.sh'.format(script_id))
    subprocess.getoutput('rm e_{}.sh'.format(script_id))
