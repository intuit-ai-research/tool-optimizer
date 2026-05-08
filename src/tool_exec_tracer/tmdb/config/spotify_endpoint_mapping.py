"""
Spotify API endpoint mapping configuration.

This module defines the mapping between API function names and their corresponding Spotify endpoints.
Used across multiple components to ensure consistency.
"""

SPOTIFY_ENDPOINT_MAPPING = {
    'add_to_queue': '/me/player/queue',
    'add_tracks_to_playlist': '/playlists/{playlist_id}/tracks',
    'change_playlist_details': '/playlists/{playlist_id}',
    'create_playlist': '/users/{user_id}/playlists',
    'follow_artists_users': '/me/following',
    'get_a_list_of_current_users_playlists': '/me/playlists',
    'get_a_users_available_devices': '/me/player/devices',
    'get_an_album': '/albums/{id}',
    'get_an_albums_tracks': '/albums/{id}/tracks',
    'get_an_artist': '/artists/{id}',
    'get_an_artists_albums': '/artists/{id}/albums',
    'get_an_artists_related_artists': '/artists/{id}/related-artists',
    'get_an_artists_top_tracks': '/artists/{id}/top-tracks',
    'get_current_users_profile': '/me',
    'get_followed': '/me/following',
    'get_information_about_the_users_current_playback': '/me/player',
    'get_new_releases': '/browse/new-releases',
    'get_playlist': '/playlists/{playlist_id}',
    'get_playlists_tracks': '/playlists/{playlist_id}/tracks',
    'get_queue': '/me/player/queue',
    'get_recently_played': '/me/player/recently-played',
    'get_recommendations': '/recommendations',
    'get_the_users_currently_playing_track': '/me/player/currently-playing',
    'get_track': '/tracks/{id}',
    'get_users_saved_albums': '/me/albums',
    'get_users_saved_tracks': '/me/tracks',
    'get_users_top_artists_and_tracks': '/me/top/{type}',
    'pause_a_users_playback': '/me/player/pause',
    'remove_albums_user': '/me/albums',
    'remove_tracks_playlist': '/playlists/{playlist_id}/tracks',
    'remove_tracks_user': '/me/tracks',
    'save_albums_user': '/me/albums',
    'save_tracks_user': '/me/tracks',
    'search': '/search',
    'set_repeat_mode_on_users_playback': '/me/player/repeat',
    'set_volume_for_users_playback': '/me/player/volume',
    'skip_users_playback_to_next_track': '/me/player/next',
    'skip_users_playback_to_previous_track': '/me/player/previous',
    'start_a_users_playback': '/me/player/play',
    'unfollow_artists_users': '/me/following',
}
