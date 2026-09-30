# POL-REFUND-DELAYED

Version: 2026-01
Category: refund

A delayed-delivery refund is a full simulated refund only when the order status is
`delayed`. Gold customers require a delay strictly greater than 7 days. Standard
customers require a delay strictly greater than 14 days. The amount is the
recorded order amount. Approval is always required; amounts above EUR 500 require
a manager, while EUR 500 exactly requires an operator.

Example: a Gold customer with a 10-day delayed EUR 199.00 order is eligible.
Example: a Standard customer with a 10-day delayed EUR 199.00 order is ineligible.