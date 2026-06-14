-- ALTER migration to add new constraints and fields to existing orders and purchases tables
-- PostgreSQL compatible version
-- Run this if you already have orders/purchases tables and want to apply the new changes
-- without dropping existing data

-- 1. Add unique constraint to orders.razorpay_order_id
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = '_razorpay_order_id_uc' AND table_name = 'orders') THEN
        ALTER TABLE orders ADD CONSTRAINT _razorpay_order_id_uc UNIQUE (razorpay_order_id);
    END IF;
END $$;

-- 2. Make purchases.razorpay_signature nullable
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'purchases' AND column_name = 'razorpay_signature' AND is_nullable = 'NO') THEN
        ALTER TABLE purchases ALTER COLUMN razorpay_signature DROP NOT NULL;
    END IF;
END $$;

-- 3. Add purchases.webhook_signature column
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'purchases' AND column_name = 'webhook_signature') THEN
        ALTER TABLE purchases ADD COLUMN webhook_signature VARCHAR(255);
    END IF;
END $$;

-- 4. Add unique constraint to purchases.order_id
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = '_order_id_uc' AND table_name = 'purchases') THEN
        ALTER TABLE purchases ADD CONSTRAINT _order_id_uc UNIQUE (order_id);
    END IF;
END $$;

-- 5. Add unique constraint to purchases.razorpay_payment_id
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = '_razorpay_payment_id_uc' AND table_name = 'purchases') THEN
        ALTER TABLE purchases ADD CONSTRAINT _razorpay_payment_id_uc UNIQUE (razorpay_payment_id);
    END IF;
END $$;
