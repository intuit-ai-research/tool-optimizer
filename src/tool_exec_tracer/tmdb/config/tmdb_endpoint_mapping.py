"""
TMDB API endpoint mapping configuration.

This module defines the mapping between API function names and their corresponding TMDB endpoints.
Used across multiple components to ensure consistency.
"""

TMDB_ENDPOINT_MAPPING = {
    # Search APIs
    'search_movies': '/search/movie',
    'search_tv': '/search/tv',
    'search_people': '/search/person',
    'search_collections': '/search/collection',
    'search_companies': '/search/company',

    # Movie APIs
    'get_movie_details': '/movie/{movie_id}',
    'get_movie_credits': '/movie/{movie_id}/credits',
    'get_movie_images': '/movie/{movie_id}/images',
    'get_movie_keywords': '/movie/{movie_id}/keywords',
    'get_movie_recommendations': '/movie/{movie_id}/recommendations',
    'get_movie_release_dates': '/movie/{movie_id}/release_dates',
    'get_movie_reviews': '/movie/{movie_id}/reviews',
    'get_similar_movies': '/movie/{movie_id}/similar',
    'get_popular_movies': '/movie/popular',
    'get_top_rated_movies': '/movie/top_rated',
    'get_upcoming_movies': '/movie/upcoming',
    'get_now_playing_movies': '/movie/now_playing',
    'get_latest_movie': '/movie/latest',
    'get_movie_genres': '/genre/movie/list',

    # TV Show APIs
    'get_tv_details': '/tv/{tv_id}',
    'get_tv_credits': '/tv/{tv_id}/credits',
    'get_tv_images': '/tv/{tv_id}/images',
    'get_tv_keywords': '/tv/{tv_id}/keywords',
    'get_tv_recommendations': '/tv/{tv_id}/recommendations',
    'get_tv_reviews': '/tv/{tv_id}/reviews',
    'get_similar_tv': '/tv/{tv_id}/similar',
    'get_popular_tv': '/tv/popular',
    'get_top_rated_tv': '/tv/top_rated',
    'get_tv_airing_today': '/tv/airing_today',
    'get_tv_on_the_air': '/tv/on_the_air',
    'get_latest_tv': '/tv/latest',
    'get_tv_genres': '/genre/tv/list',

    # TV Season & Episode APIs
    'get_tv_season_details': '/tv/{tv_id}/season/{season_number}',
    'get_tv_season_credits': '/tv/{tv_id}/season/{season_number}/credits',
    'get_tv_season_images': '/tv/{tv_id}/season/{season_number}/images',
    'get_tv_episode_details': '/tv/{tv_id}/season/{season_number}/episode/{episode_number}',
    'get_tv_episode_credits': '/tv/{tv_id}/season/{season_number}/episode/{episode_number}/credits',
    'get_tv_episode_images': '/tv/{tv_id}/season/{season_number}/episode/{episode_number}/images',

    # People APIs
    'get_person_details': '/person/{person_id}',
    'get_person_images': '/person/{person_id}/images',
    'get_person_movie_credits': '/person/{person_id}/movie_credits',
    'get_person_tv_credits': '/person/{person_id}/tv_credits',
    'get_popular_people': '/person/popular',

    # Discover APIs
    'discover_movies': '/discover/movie',
    'discover_tv': '/discover/tv',

    # Trending APIs
    'get_trending': '/trending/{media_type}/{time_window}',
    'get_trending_movies': '/trending/movie/{time_window}',
    'get_trending_tv': '/trending/tv/{time_window}',
    'get_trending_all': '/trending/all/{time_window}',

    # Collection APIs
    'get_collection_details': '/collection/{collection_id}',
    'get_collection_images': '/collection/{collection_id}/images',

    # Company APIs
    'get_company_details': '/company/{company_id}',
    'get_company_images': '/company/{company_id}/images',

    # Network APIs
    'get_network_details': '/network/{network_id}',
    'get_network_images': '/network/{network_id}/images',

    # Misc APIs
    'get_credit_details': '/credit/{credit_id}',
    'get_review_details': '/review/{review_id}',
}