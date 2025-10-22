from nnunetv2.configuration import default_num_processes
from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess
import argparse
def nnUNetv2_preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+', type=int,
                        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                             "planning and preprocessing for these datasets. Can of course also be just one dataset")
    parser.add_argument('-plans_name', default='nnUNetPlans', required=False,
                        help='[OPTIONAL] You can use this to specify a custom plans file that you may have generated')
    parser.add_argument('-c', required=False, default=['2d', '3d_fullres', '3d_lowres'], nargs='+',
                        help='[OPTIONAL] Configurations for which the preprocessing should be run. Default: 2d 3d_fullres '
                             '3d_lowres. 3d_cascade_fullres does not need to be specified because it uses the data '
                             'from 3d_fullres. Configurations that do not exist for some dataset will be skipped.')
    parser.add_argument('-np', type=int, nargs='+', default=None, required=False,
                        help="[OPTIONAL] Use this to define how many processes are to be used. If this is just one number then "
                             "this number of processes is used for all configurations specified with -c. If it's a "
                             "list of numbers this list must have as many elements as there are configurations. We "
                             "then iterate over zip(configs, num_processes) to determine then umber of processes "
                             "used for each configuration. More processes is always faster (up to the number of "
                             "threads your PC can support, so 8 for a 4 core CPU with hyperthreading. If you don't "
                             "know what that is then dont touch it, or at least don't increase it!). DANGER: More "
                             "often than not the number of processes that can be used is limited by the amount of "
                             "RAM available. Image resampling takes up a lot of RAM. MONITOR RAM USAGE AND "
                             "DECREASE -np IF YOUR RAM FILLS UP TOO MUCH!. Default: 8 processes for 2d, 4 "
                             "for 3d_fullres, 8 for 3d_lowres and 4 for everything else")
    parser.add_argument('--verbose', required=False, action='store_true',
                        help='Set this to print a lot of stuff. Useful for debugging. Will disable progress bar! '
                             'Recommended for cluster environments')
    args, unrecognized_args = parser.parse_known_args()

    args.c='2d'
    args.d=[2,]
    args.plans_name="nnUNetResEncUNetMPlans"
    if args.np is None:
        default_np = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}
        np = [default_np[c] if c in default_np.keys() else 4 for c in args.c]
    else:
        np = args.np


    preprocess(args.d, args.plans_name, configurations=args.c, num_processes=np, verbose=args.verbose)
if __name__=='__main__':
    nnUNetv2_preprocess()