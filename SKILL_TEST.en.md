# Background

Suppose you are assigned to a project that develops a price prediction service for an online marketplace. You will build a REST service which returns a predicted price from listing information.

# Assignment

Develop a price prediction model and a REST endpoint

## POST /v1/price

### Request body

- Any of item information listed in the Data section
- e.g.

```
{
  "name":"Hold Alyssa Frye Harness boots 12R, Sz 7",
  "item_condition_id":3,
  "category_name":"Women/Shoes/Boots",
}
```

### Response

- Json format
- e.g. when a predicted price is $30, it should be `{"price": 30}`.

# Data

The data is available in the `data/` directory.
`mercari_train.csv` and `mercari_test.csv` consist of a list of product listings

- `id`: the id of the listing
- `name`: the title of the listing
- `item_condition_id`: the condition of the items provided by the seller
- `category_name`: category of the listing
- `brand_name`: brand of the listing
- `price`: the price (USD) that the item was sold for. This column doesn't exist in `mercari_test.csv`
- `shipping`: 1 if shipping fee is paid by seller and 0 by buyer
- `item_description`: the full description of the listing
- `seller_id`: the seller ID of the listing

# Requirements

- Python 3.7 or higher
- A Dockerfile is required and your API server should be runnable on Docker
- Unit tests for the API server
- A README which describes how to run the unit test and server
- Training code for your price prediction model
- Model file size should be smaller than 10MB (You can compress your model file)
- A csv file for your prediction result (Please refer to `sample_submission.csv`)
- Root mean squared logarithmic error (RMSLE) for the `mercari_test.csv` should be less than 0.5