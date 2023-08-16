import torch
from torch.utils.data import Dataset, Subset, DataLoader

def get_predictions(data: Dataset = None,
                   loader: DataLoader = None,
                   test_indices: list[int] = None,
                   predictor: torch.nn.Module = None,
                   cfg: dict = None):
    '''
        Apply the predictor to the test loader
    '''
    predictor.eval()

    # If the data loader is not provided, create a data loader from the data.
    if loader is None:
        assert data is not None, "data and loader cannot both be None"

        if test_indices is None:
            # Use the entire dataset for evaluation.
            test_data = data

        else:
            # Create the testing split dataset by selecting the subset of the
            # original dataset.
            test_data = Subset(data, indices=test_indices)

        loader = DataLoader(test_data, batch_size=cfg['batch_size'],
                            shuffle=False, num_workers=cfg['num_workers'])


    with torch.no_grad():

        results = {'y': [], 'pred_y': []}
        i = 0

        for x, y in loader:

            x = x.to(cfg['device'])
            y = y.to(cfg['device'])

            out = predictor(x)
            print(out)

            # Compile the results
            pred_y = torch.argmax(out, dim=1).cpu().numpy().tolist()
            y = y.cpu().numpy().tolist()
            results["y"].extend(y)
            results["pred_y"].extend(pred_y)
            i += 1 

    return results