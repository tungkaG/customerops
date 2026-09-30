# POL-ADDRESS-CHANGE

Version: 2026-01
Category: address

Only an order with status `processing` can receive a shipping-address change.
The request must provide line1, city, postal_code, and a two-letter country code.
Operator approval is required before the address changes.