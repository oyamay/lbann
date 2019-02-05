# Global imports
import os
import sys
import json
import matplotlib.pyplot as plt
import texttable as tt

# Local imports
from . import parser

def _get_time_axis(time_list, units='hours'):
    """Convert time to sequential format and convert time units."""
    time_axis = []
    for i in range(len(time_list)):
        time_sum = sum(time_list[:i])
        if units == 'seconds':
            pass
        elif units == 'minutes':
            time_sum /= 60.0
        elif units == 'hours':
            time_sum /= 3600.0
        time_axis.append(time_sum)
    return time_axis

def plot(stat_path_list, stat_name_list, ind_var='time', time_units='hours', plot_accuracy=True, merge_train_val=False):
    """Tabulate and plot stats from LBANN or PyTorch training in common format."""
    ### Load stat dicts and print stat summary
    stat_dict_list = []
    # Get run names
    if stat_name_list is None:
        print('WARNING: No trial names provided, using file names by default.')
        run_name_list = [os.path.splitext(os.path.basename(stat_path))[0] for stat_path in stat_path_list]
    elif len(stat_name_list) != len(stat_path_list):
        print('WARNING: # stat paths ({}) does not match # stat names ({}). Using file names by default.'.format(
            len(stat_path_list), len(stat_name_list)))
        run_name_list = [os.path.splitext(os.path.basename(stat_path))[0] for stat_path in stat_path_list]
    else:
        run_name_list = stat_name_list
    # Create table for comparing trials
    stat_table = tt.Texttable()
    headings = ['Trial', 'Num Epochs', 'Avg. Train Time (s)', 'Avg. Val Time (s)']
    if plot_accuracy:
        headings += ['Peak Train Acc', 'Peak Val Acc']

    stat_table.header(headings)
    # Loop through each trial
    for run_name, stat_path in zip(run_name_list, stat_path_list):
        # Load stat file
        stat_ext = os.path.splitext(stat_path)[1]
        if stat_ext == '.json':
            with open(stat_path, 'r') as fp:
                d = json.load(fp)
        elif stat_ext == '.out' or stat_ext == '.txt':
            d = parser.parse(stat_path)
        else:
            print('ERROR: Invalid file extension: {} from {}\nPlease provide either an LBANN output file with .out or .txt extension or a PyTorch output file with .json extension.'.format(stat_ext, stat_path))
            sys.exit(1)

        # Total epochs of training
        total_epochs = len(d['val_time'])

        # Compute accuracy stats
        if plot_accuracy:
            peak_train_acc = max(d['train_acc'])
            peak_train_epoch = d['train_acc'].index(peak_train_acc)
            peak_val_acc = max(d['val_acc'])
            peak_val_epoch = d['val_acc'].index(peak_val_acc)

        # Compute loss stats
        min_train_loss = min(d['train_loss'])
        min_train_epoch = d['train_loss'].index(min_train_loss)
        min_val_loss = min(d['val_loss'])
        min_val_epoch = d['val_loss'].index(min_val_loss)

        # Compute time stats
        avg_train_time = int(sum(d['train_time'])/len(d['train_time']))
        avg_val_time = int(sum(d['val_time'])/len(d['val_time']))

        # Create independent variable axis
        if ind_var == 'epoch':
            d['train_axis'] = range(len(d['train_time']))
            d['val_axis'] = range(len(d['val_time']))
            xlabel = 'Epoch'
        elif ind_var == 'time':
            d['train_axis'] = _get_time_axis(d['train_time'], units=time_units)
            d['val_axis'] = _get_time_axis(d['val_time'], units=time_units)
            xlabel = 'Time ({})'.format(time_units)
        else:
            raise Exception('Invalid indepedent variable: {}'.format(ind_var))

        # Store the stat dict for plotting
        stat_dict_list.append((run_name, d))

        # Add row to stats table for current trial
        # stat_table.add_row([run_name, total_epochs, avg_train_time, avg_val_time, peak_train_acc, peak_val_acc]) # FIXME
        stat_table.add_row([run_name, total_epochs, avg_train_time, avg_val_time] \
                           + ([peak_train_acc, peak_val_acc] if plot_accuracy else []))

    # Print the stats table
    print()
    table_str = stat_table.draw()
    print(table_str)
    print()

    ### Plot stats
    plt.figure(figsize=(12, 10 if plot_accuracy else 5))
    plt.suptitle('Trial Stats vs. {}'.format(ind_var.title(), fontsize=20))
    subplot_nrow = 2 if plot_accuracy else 1
    subplot_ncol = 2 if not merge_train_val else 1
    for run_name, stat_dict in stat_dict_list:
        run_name_train = run_name if not merge_train_val else run_name+' (train)'
        run_name_val   = run_name if not merge_train_val else run_name+' (val)'

        # Train acc
        if plot_accuracy:
            plt.subplot(subplot_nrow, subplot_ncol, 1)
            plt.title('Train Accuracy vs. {}'.format(ind_var.title()))
            plt.xlabel(xlabel)
            plt.ylabel('Train Accuracy')
            plt.plot(stat_dict['train_axis'], stat_dict['train_acc'], label=run_name_train)

        # Val acc
        if plot_accuracy:
            plt.subplot(subplot_nrow, subplot_ncol, 2 if not merge_train_val else 1)
            plt.title('Val Accuracy vs. {}'.format(ind_var.title()))
            plt.xlabel(xlabel)
            plt.ylabel('Val Accuracy')
            plt.plot(stat_dict['val_axis'], stat_dict['val_acc'], label=run_name_val)

        loss_subplot = 1 + ((2 if not merge_train_val else 1) if plot_accuracy else 0)

        # Train loss
        plt.subplot(subplot_nrow, subplot_ncol, loss_subplot)
        plt.title('Train Loss vs. {}'.format(ind_var.title()))
        plt.xlabel(xlabel)
        plt.ylabel('Train Loss')
        p, = plt.plot(stat_dict['train_axis'], stat_dict['train_loss'], label=run_name_train)

        # Val loss
        plt.subplot(subplot_nrow, subplot_ncol, loss_subplot + (1 if not merge_train_val else 0))
        plt.title('Val Loss vs. {}'.format(ind_var.title()))
        plt.xlabel(xlabel)
        plt.ylabel('Val Loss')
        kwargs = {} if not merge_train_val else {"color": p.get_color(), "linestyle": "dashed"}
        plt.plot(stat_dict['val_axis'], stat_dict['val_loss'], label=run_name_val, **kwargs)

    # Legend position will likely only be good for the test example
    # plt.legend(loc=(0.25, 1.22))
    plt.legend()
    # Show the plot
    plt.show()
