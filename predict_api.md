# Prediction API definition

## Input features

| Field | Description
| --- | --- |
| brand | Brand id hash (e.g. "b25", "b124")
| category-1 | Product level 1 category (e.g. "clothing", "womenswear", "menswear", "kidswear" or "home")
| category-2 | Product level 2 category (e.g. "outerwear", "thermals", "swimwear")
| category-3 | Product level 3 category (e.g. "knitwear", "coats", "jacket", "skirts", "trousers")
| colour | Product colour
| fabric_type | Product fabric type (e.g. "knit", "woven")
| ftp_acrylic | Fibre type percentage of Acrylic
| ftp_cotton | Fibre type percentage of Cotton
| ftp_elastane | Fibre type percentage of Elastene
| ftp_linen | Fibre type percentage of Linen
| ftp_polyamide | Fibre type percentage of Polyamide
| ftp_polyester | Fibre type percentage of Polyester
| ftp_polypropylene | Fibre type percentage of Polypropylene
| ftp_silk | Fibre type percentage of Silk
| ftp_viscose | Fibre type percentage of Viscose
| ftp_wool | Fibre type percentage of Wool
| ftp_other | Fibre type percentage of other materials
| gender | Gender code, see gender mappings below
| label | Product label if we have one
| made_in | Made in country, see ISO 3166-1 alpha-2 (e.g. "FI", "GR")
| season | Season code, see season mappigs below (e.g. "MID", "SUM")
| size | Product size (e.g. "XS", "S", "M", "L", "XL" etc)
| unspsc_code | Product or service UNSPSC code, if known
| weight | Products weight in kilograms

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
|  All-year round | AYR
| winter | WIN"
| midterm | MID"
| summer | SUM"

## Output
| Field | Description
| --- | --- |
| co2_total | Amount of kgCO2e / product.