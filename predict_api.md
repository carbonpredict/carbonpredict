# Prediction API definition

Documented also using OpenAPI (version 2), UI available at http://*hostname*:5000/apidocs/.

## Endpoint
*/ccaas/api/v0.1/predict* (POST).

HTTP body is type application/json and contains the input features in JSON format (see below).

## Returns
HTTP 201 with message body of type application/json containing prediction (estimated kilograms of CO2e for product) and intervals. The intervals are null if they are not available for the requested model

Example response body:
```
{
  "prediction": 2.8923212,
  "5-percentile": 1.2988872,
  "95-percentile": 6.610098 
}
```

## Example message body
```
{
  "ML-model": "lgbm_default",
  "brand": "b119",
  "category-1": "kidswear",
  "category-2": "suit",
  "category-3": "tie",
  "colour": "blue gray",
  "fabric_type": "K",
  "ftp_acrylic": 15,
  "ftp_cotton": 0,
  "ftp_elastane": 0,
  "ftp_linen": 33,
  "ftp_other": 7,
  "ftp_polyamide": 0,
  "ftp_polyester": 14,
  "ftp_polypropylene": 11,
  "ftp_silk": 15,
  "ftp_viscose": 0,
  "ftp_wool": 10,
  "gender": "Y",
  "label": {},
  "made_in": "CN",
  "season": "AYR",
  "size": "XS",
  "unspsc_code": {},
  "weight": 0.1
}
```

## Input features

| Field | Type | Description
| --- | --- | --- |
| ML-model | string | Machine learning model name to use for prediction (e.g. lgbm_default)
| brand | string | Brand id hash (e.g. "b25", "b124")
| category-1 | string | Product level 1 category (e.g. "clothing", "womenswear", "menswear", "kidswear" or "home")
| category-2 | string | Product level 2 category (e.g. "outerwear", "thermals", "swimwear")
| category-3 | string | Product level 3 category (e.g. "knitwear", "coats", "jacket", "skirts", "trousers")
| colour | string | Product colour
| fabric_type | number | Product fabric type, K = "knit" or W = "woven")
| ftp_acrylic | number | Fibre type percentage of Acrylic
| ftp_cotton | number | Fibre type percentage of Cotton
| ftp_elastane | number | Fibre type percentage of Elastene
| ftp_linen | number | Fibre type percentage of Linen
| ftp_other | number | Fibre type percentage of other materials
| ftp_polyamide | number | Fibre type percentage of Polyamide
| ftp_polyester | number | Fibre type percentage of Polyester
| ftp_polypropylene | number | Fibre type percentage of Polypropylene
| ftp_silk | number | Fibre type percentage of Silk
| ftp_viscose | number | Fibre type percentage of Viscose
| ftp_wool | number | Fibre type percentage of Wool
| gender | string | Gender code, see gender mappings below
| label | string | Product label, NOT IN USE
| made_in | string | Made in country, see ISO 3166-1 alpha-2 (e.g. "FI", "GR")
| season | string | Season code, see season mappings below (e.g. "MID", "SUM")
| size | string | Product size (e.g. "XS", "S", "M", "L", "XL" etc)
| unspsc_code | string | Product or service UNSPSC code, NOT IN USE
| weight | number | Product weight in kilograms

### Gender mappings

| Gender | Code
| --- | --- |
| Unisex | U
| Women | W
| Men | M
| Kids | K
| Girls | G
| Boys | B
| Baby | Y

### Season mappings

| Gender | Code
| --- | --- |
| All-year round | AYR
| winter | WIN
| midterm | MID
| summer | SUM