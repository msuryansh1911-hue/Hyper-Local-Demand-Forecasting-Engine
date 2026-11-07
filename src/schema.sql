DROP VIEW IF EXISTS v_training;
DROP TABLE IF EXISTS sales_daily;
DROP TABLE IF EXISTS weather_daily;
DROP TABLE IF EXISTS holidays;

CREATE TABLE weather_daily (
    date        TEXT PRIMARY KEY,   -- ISO yyyy-mm-dd
    temp_max    REAL NOT NULL,
    temp_min    REAL NOT NULL,
    precip_mm   REAL NOT NULL
);

CREATE TABLE holidays (
    date        TEXT PRIMARY KEY,
    name        TEXT NOT NULL
);

CREATE TABLE sales_daily (
    date        TEXT NOT NULL,
    category    TEXT NOT NULL,
    units_sold  INTEGER NOT NULL,
    PRIMARY KEY (date, category),
    FOREIGN KEY (date) REFERENCES weather_daily(date)
);

CREATE INDEX idx_sales_category ON sales_daily(category);

CREATE VIEW v_training AS
SELECT s.date, s.category, s.units_sold,
       w.temp_max, w.temp_min, w.precip_mm,
       CASE WHEN h.date IS NULL THEN 0 ELSE 1 END AS is_holiday,
       h.name AS holiday_name
FROM sales_daily s
JOIN weather_daily w ON w.date = s.date
LEFT JOIN holidays h ON h.date = s.date;
