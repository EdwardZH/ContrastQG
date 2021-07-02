import os
import sys
import time
import logging
import argparse

logger = logging.getLogger()

# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
def add_default_args(parser):
    
    ## ************************
    # Modes
    ## ************************
    modes = parser.add_argument_group("Modes")
    modes.add_argument(
        "--no_cuda", 
        action="store_true", 
        default=False,
        help="Train model on GPUs.",
    )
    modes.add_argument(
        "--data_workers", 
        default=0, 
        type=int, 
        help="Number of subprocesses for data loading",
    )
    modes.add_argument(
        "--seed", 
        default=42, 
        type=int, 
        help="Random seed for initialization: 42",
    )
    modes.add_argument(
        "--task",
        type=str,
        help="Ranking Tasks",
    )
    
    ## ************************
    # File
    ## ************************
    files = parser.add_argument_group("Files")
    files.add_argument("--output_dir",
                       required=True,
                       type=str,
                       help="Target dataset path",
                      )
    files.add_argument("--input_dir",
                       required=True,
                       type=str,
                       help="Raw dataset path",
                      )
    modes.add_argument(
        "--save_txt",
        action="store_true",
        default=False,
        help="Save tsv files.",
    )
    ## ************************
    # Generator
    ## ************************
    generator = parser.add_argument_group("Generator")
    
    generator.add_argument(
        "--generator_mode", 
        choices=["cqg", "qg"],
        required=True,
        type=str, 
        help="Select contrastqg or qg mode",
    )

    generator.add_argument(
        "--do_sample",
        action="store_true",
        default=False,
        help="do sampling.",
    )

    generator.add_argument(
        "--pretrain_generator_type", 
        choices=["t5-small", "t5-base"],
        default="t5-small",
        type=str,
        help="Select pretrain generator type.",
    )
    generator.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for test."
    )
    generator.add_argument(
        "--model_dir",
        type=str, 
        required=True
    )
    generator.add_argument(
        "--max_input_len",
        type=int, 
        default=512
    )
    generator.add_argument(
        "--max_qg_doc",
        type=int,
        default=200000
    )
    generator.add_argument(
        "--max_gen_len", 
        type=int, 
        default=32, 
        help="Maximum length of output sequence"
    )
    generator.add_argument(
        "--min_gen_length", 
        type=int, 
        default=20
    )
    generator.add_argument(
        "--top_k",
        type=float, 
        default=25,
        help="topk sampling"
    )
    generator.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="the number of return sequences"
    )
    generator.add_argument(
        "--top_p", 
        type=float, 
        default=0.95,
        help="The cumulative probability of parameter highest probability \
        vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1."
    )
    generator.add_argument(
        "--retry_times",
        type=int,
        default=3
    )

    
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
def init_args_config(args):
    
    # logging file
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO) # logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s: [ %(message)s ]", "%m/%d/%Y %I:%M:%S %p")
    
    console = logging.StreamHandler() 
    console.setFormatter(fmt) 
    logger.addHandler(console) 
    logger.info("COMMAND: %s" % " ".join(sys.argv))