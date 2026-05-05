# Unseen Tool Generalization

Split: `all`.

Definition: A native unseen-tool row follows the benchmark has_unseen_tool label and uses tool_split=test_unseen tools as the target subset.

- Train tasks: 4867
- Train gold tools: 497
- Train gold tool families: 259
- Eval tasks: 1998
- Eval non-empty workflows: 1303
- Eval unseen tool ids: 51
- Eval unseen tool families: 45
- Unseen source counts: {'tau2': 2, 'toolbench': 70}
- Unseen domain counts: {'airline': 1, 'mock': 1, 'validation': 70}

| method | overall n | overall tool EM | seen n | seen tool EM | unseen n | unseen tool EM | unseen schema valid | unseen tool recall | unseen family recall | unseen avg ED |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FM nearest | 1303 | 0.2671 | 1231 | 0.2827 | 72 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.0139 |
| FM constrained nearest | 1303 | 0.3200 | 1231 | 0.3266 | 72 | 0.2083 | 0.8472 | 0.5542 | 0.6933 | 1.4583 |
| FM AR available-mask | 1303 | 0.3208 | 1231 | 0.3225 | 72 | 0.2917 | 0.7778 | 0.5783 | 0.5867 | 1.4167 |
| FM optimized constrained | 1303 | 0.3101 | 1231 | 0.3136 | 72 | 0.2500 | 0.8472 | 0.5663 | 0.6800 | 1.5556 |

Interpretation: this is a workflow-planning generalization check, separate from tau2 closed-loop execution. Tool EM measures full workflow equality; unseen tool/family recall measures whether the decoder at least selects the held-out schema element; schema valid measures whether predicted tools stay inside the task's available-tool catalog.

Representative unseen tool ids:

- `tau2::airline::book_reservation`
- `tau2::mock::transfer_to_human_agents`
- `toolbench::validation::1.`
- `toolbench::validation::all_belgian_races_for_wonderboyapi`
- `toolbench::validation::all_for_coupons`
- `toolbench::validation::balance_sheet_statement_for_financial_statements`
- `toolbench::validation::blacklist_phone_numbers_for_smsapi_com`
- `toolbench::validation::carbonfootprintfrommotorbike_for_carbonfootprint`
- `toolbench::validation::date_and_time_of_last_update_for_wonderboyapi`
- `toolbench::validation::details_for_patreon`
- `toolbench::validation::downloadfile_for_aspose_tasks_cloud`
- `toolbench::validation::enneagram_personality_test_questionnaire_for_personality_quest`
- `toolbench::validation::estimate_market_value_for_car_utils`
- `toolbench::validation::findalllistingactive_for_etsy`
- `toolbench::validation::get_a_list_of_domains_for_url_link_shortener`
- `toolbench::validation::get_all_climate_change_news_for_climate_change_live_api`
- `toolbench::validation::get_hebrew_month_and_date_for_enoch_calendar`
- `toolbench::validation::get_markets_for_coinranking`
- `toolbench::validation::get_novel_by_id_for_anime_manga_and_novels_api`
- `toolbench::validation::get_type_of_place_filters_for_airbnb_v2`
- `toolbench::validation::get_user_by_gender_for_fake_users`
- `toolbench::validation::get_word_of_the_day_from_multiple_sources_for_word_of_the_day`
- `toolbench::validation::getcompanies_for_get_360_business_tool`
- `toolbench::validation::getinterestinglistings_for_etsy`
- `toolbench::validation::gettag_for_sms_receive`
- `toolbench::validation::getupdates_for_sms_receive`
- `toolbench::validation::gst_for_gst_advance`
- `toolbench::validation::healthcheck_for_hapihub`
- `toolbench::validation::hello_for_hello_world_v2`
- `toolbench::validation::image_search_for_bing_image_search`

Representative unseen tool families:

- `tau2::airline::book`
- `tau2::mock::transfer_to_human_agents`
- `toolbench::validation::1.`
- `toolbench::validation::airbnb_v2`
- `toolbench::validation::anime_manga_and_novels_api`
- `toolbench::validation::apfelpreise`
- `toolbench::validation::aspose_tasks_cloud`
- `toolbench::validation::bing_image_search`
- `toolbench::validation::bravenewcoin`
- `toolbench::validation::car_utils`
- `toolbench::validation::carbonfootprint`
- `toolbench::validation::city_boundary_by_name_for_de_boundaries_io`
- `toolbench::validation::climate_change_live_api`
- `toolbench::validation::coinranking`
- `toolbench::validation::coupons`
- `toolbench::validation::currencyapi_net`
- `toolbench::validation::direct_wines`
- `toolbench::validation::enoch_calendar`
- `toolbench::validation::etsy`
- `toolbench::validation::fake_users`
- `toolbench::validation::famous_quotes`
- `toolbench::validation::financial_statements`
- `toolbench::validation::get_360_business_tool`
- `toolbench::validation::gst_advance`
- `toolbench::validation::hapihub`
- `toolbench::validation::hello_world_v2`
- `toolbench::validation::ip2proxy`
- `toolbench::validation::mantis_object_detection`
- `toolbench::validation::mymemory_translation_memory`
- `toolbench::validation::patreon`
