from pyepo.data import knapsack

def generate_data(num_items: int):
    # TODO: Miquel's data generation implementation
    weights, features, values = knapsack.genData(
        num_data=1,
        num_features=2*num_items,
        num_items=num_items
    )
    
    return weights, features, values

weights, features, values = generate_data(num_items=20)

