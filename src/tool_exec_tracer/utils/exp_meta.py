import argparse
import subprocess
import os
import sys

LOG_TEMPLATE="""
# Reproducibility Log

## Git Cleanness
{git_cleanness}

## Git Commit Hash
{git_commit_hash}

## Command Line Arguments
{command_line_arguments}

## Environment Variables
{environment_variables}

## Args
{args}
"""

# Build log for git cleaness, git commit hash, command line arguments, environment variables, args.
def build_reproducibility_log(args: argparse.Namespace) -> str:
    git_cleanness = get_git_cleanness()
    git_commit_hash = get_git_revision_short_hash()
    command_line_arguments = " ".join(sys.argv)
    environment_variables = os.environ
    args = args
    return LOG_TEMPLATE.format(git_cleanness=git_cleanness, git_commit_hash=git_commit_hash, command_line_arguments=command_line_arguments, environment_variables=environment_variables, args=args)

def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

def get_git_cleanness() -> str:
    return subprocess.check_output(['git', 'status', '--porcelain']).decode('ascii').strip()