/**
 * The boot-time `/api/**` answers every screenshot harness needs.
 *
 * A capture script renders the REAL built SPA, so the shell fetches its usual
 * startup endpoints before the page under test appears: prerequisites, slots,
 * theme, branding, notifications. None of them are what a harness is
 * photographing, and a 404 on any of them can keep the page from rendering at
 * all -- so every harness answers the same set, and until now each one carried its
 * own copy of the block.
 *
 * `bootApiStub` returns a handler for the FRAMEWORK endpoints only. A harness
 * passes its own `routes` map for the endpoints its scene is actually about, and
 * those win: the stub is the floor, never an override.
 */

/** Fulfil a Playwright route with a JSON body. */
export const json = (route, body, status = 200) => route.fulfill({
  status, contentType: 'application/json', body: JSON.stringify(body),
})

/**
 * A `page.route('**\/api\/**', ...)` handler.
 *
 * @param {object} [options]
 * @param {Record<string, unknown>} [options.routes] pathname -> body, checked first.
 * @param {'light'|'dark'} [options.mode] what `/api/theme/boot` should report.
 */
export function bootApiStub({ routes = {}, mode = 'dark' } = {}) {
  return async (route) => {
    const path = new URL(route.request().url()).pathname

    if (Object.prototype.hasOwnProperty.call(routes, path)) {
      return json(route, routes[path])
    }

    if (path === '/api/theme/boot') return json(route, { mode, theme: '' })
    if (path === '/api/kiro-prerequisite') {
      return json(route, {
        platform: 'linux', installed: true, authenticated: true, ready: true,
        initial_setup_complete: true, can_auto_install: false, can_login: false,
        repair_required: false, docs_url: '', setup_allowed: false,
        operation: { kind: '', status: 'idle', message: '', detail: '', url: '', error: '' },
      })
    }
    if (path === '/api/chat/slots') return json(route, [])
    if (path.startsWith('/api/instances')) return json(route, { instances: [], active: '' })
    if (path === '/api/status') {
      return json(route, { sessions: 0, crons: 0, lessons: 0, uptime: 120, version: 'dev' })
    }
    if (path === '/api/notifications') return json(route, { notifications: [], unread: 0 })
    if (path === '/api/auth/me') return json(route, { user: 'owner', app: '' })
    if (path === '/api/themes') return json(route, { themes: [], installed: [] })
    if (path === '/api/dashboard/branding') return json(route, { bot_name: 'Kiro', avatar: '' })
    if (path === '/api/dashboard/config') {
      return json(route, {
        restore_sessions: false, restore_window_minutes: 30,
        merge_queued_messages: false, widget_density: 'more',
      })
    }

    // An unknown endpoint answers an empty OBJECT when its name suggests a record
    // and an empty ARRAY otherwise, because a page that destructures a config will
    // throw on `[]` and one that maps a list will throw on `{}`.
    const objectish = /(config|tips|voice|autonudge|branding|status|usage-summary|ui-prefs)/
    return json(route, objectish.test(path) ? {} : [])
  }
}
