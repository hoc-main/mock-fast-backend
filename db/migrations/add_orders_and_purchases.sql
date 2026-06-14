-- Migration to add orders and purchases tables
-- Final production-ready version

CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    module_id INTEGER NOT NULL REFERENCES modules(id),
    razorpay_order_id VARCHAR(100) NOT NULL,
    amount INTEGER NOT NULL,
    currency VARCHAR(10) NOT NULL DEFAULT 'INR',
    status VARCHAR(50) NOT NULL DEFAULT 'created', -- created, paid, failed
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (razorpay_order_id)
);

CREATE TABLE IF NOT EXISTS purchases (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    module_id INTEGER NOT NULL REFERENCES modules(id),
    order_id INTEGER NOT NULL REFERENCES orders(id),
    razorpay_payment_id VARCHAR(100) NOT NULL,
    razorpay_order_id VARCHAR(100) NOT NULL,
    razorpay_signature VARCHAR(255),
    webhook_signature VARCHAR(255),
    amount INTEGER NOT NULL,
    currency VARCHAR(10) NOT NULL DEFAULT 'INR',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (order_id),
    UNIQUE (razorpay_payment_id)
);

-- Create indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_orders_user_id ON orders(user_id);
CREATE INDEX IF NOT EXISTS idx_orders_module_id ON orders(module_id);
CREATE INDEX IF NOT EXISTS idx_purchases_user_id ON purchases(user_id);
CREATE INDEX IF NOT EXISTS idx_purchases_module_id ON purchases(module_id);
CREATE INDEX IF NOT EXISTS idx_purchases_order_id ON purchases(order_id);
