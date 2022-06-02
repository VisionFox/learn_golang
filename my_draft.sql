create table ip_range_info(
    `id`           int AUTO_INCREMENT,
    `ip_start`     int        not null,
    `ip_end`       int        not null,
    `country_code` varchar(8) not null,
    `update_time`  timestamp  not null default current_timestamp on update current_timestamp,
    `create_time`  timestamp  not null default current_timestamp,
    PRIMARY KEY (`id`),
    UNIQUE KEY (`ip_start`),
    UNIQUE KEY `uk_ip_start_end_country_code` (`ip_start`, `ip_end`, `country_code`),
    INDEX idx_ip_country_code (`ip_start`, `ip_end`, `country_code`)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

select country_code, ip_end from ip_range_info where ip_start >= 99 order by ip_start asc limit 0,1;