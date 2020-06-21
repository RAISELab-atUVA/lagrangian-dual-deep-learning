from model import *

def main():

    dataset_names = ['default', 'census', 'bank']
    perf_dict_per_data = {}  # KEYS are default, census, bank

    for name in dataset_names:
        pd00, label_name, z_name, feats = load_data(name)
        perf_per_model = {}  # KEYS are MC, MC-1, MC-D
        print('Working with dataset = {}'.format(name))
        for key_ in ['MC', 'MC-1', 'MC-D']:
            perf_per_model[key_] = []

        best_options = {}
        if name == 'bank':
            step_size_choice = 0.02
        elif name == 'default':
            step_size_choice = 0.2
        elif name == 'census':
            step_size_choice = 0.3
        else:
            step_size_choice = 0.2
        for seed in range(5):
            X_train, X_val, X_test, y_train, y_val, y_test, Z_train, Z_val, Z_test = get_data_loader(pd00, feats,
                                                                                                     label_name, z_name,
                                                                                                     seed=seed)
            params = {}
            params['X_train'] = X_train
            params['X_val'] = X_val
            params['y_train'] = y_train
            params['y_val'] = y_val
            params['Z_train'] = Z_train
            params['Z_val'] = Z_val
            params['device'] = 'cuda'
            params['batch_size'] = 300
            options = {}
            options['model_lr'] = 1e-3
            options['step_size'] = 0
            options['lr_mult'] = 0
            options['epochs'] = 100
            input_dim = X_train.shape[1]
            options['input_size'] = input_dim
            grid_search_list = []
            mc_model = LDSharedModel(params)

            if seed == 0:
                for n_gen in [int(x) for x in np.linspace(input_dim / 2, input_dim - 1, 2)]:
                    for n_z in [int(x) for x in np.linspace(input_dim / 5 + 1, input_dim / 2, 2)]:
                        if n_z < n_gen:
                            curr_options = copy.deepcopy(options)
                            curr_options['model_params'] = {'input_size': input_dim, 'gen_nnodes_list': [n_gen],
                                                            'z1_nnodes_list': [n_z], \
                                                            'z0_nnodes_list': [n_z]}
                            grid_search_list.append(curr_options)

                mc_model.hyper_opt(grid_search_list)
                best_options = mc_model.best_options

            if len(best_options.keys()) == 0:
                print('WARNING we havent optimized our network structure')

            best_options['lr_mult'] = 0
            best_options['step_size'] = 0
            if mc_model.model is None:
                mc_model.fit(best_options)

            y_pred_mc = mc_model.predict(X_test, Z_test)[0]
            mc_acc, _, mc_di = compute_fairness_score(y_test, y_pred_mc, Z_test)
            perf_per_model['MC'].append((mc_acc, mc_di, mc_model.logs))

            mc1_model = LDSharedModel(params)
            best_options['lr_mult'] = 1.0
            best_options['step_size'] = 0
            mc1_model.fit(best_options)

            y_pred_mc1 = mc1_model.predict(X_test, Z_test)[0]
            mc1_acc, _, mc1_di = compute_fairness_score(y_test, y_pred_mc1, Z_test)
            perf_per_model['MC-1'].append((mc1_acc, mc1_di, mc1_model.logs))

            mcd_model = LDSharedModel(params)
            best_options['lr_mult'] = 0.0
            best_options['step_size'] = step_size_choice
            mcd_model.fit(best_options)

            y_pred_mcd = mcd_model.predict(X_test, Z_test)[0]
            mcd_acc, _, mcd_di = compute_fairness_score(y_test, y_pred_mcd, Z_test)
            perf_per_model['MC-D'].append((mcd_acc, mcd_di, mcd_model.logs))

        perf_dict_per_data[name] = {'MC': perf_per_model['MC'], 'MC-1': perf_per_model['MC-1'],
                                    'MC-D': perf_per_model['MC-D'], 'best_options': best_options}


    file_handle = open('ECML_100eps_300bs_adaptive_step_size.pkl','wb')
    pickle.dump(perf_dict_per_data, file_handle)


if __name__ == '__main__':
    main()